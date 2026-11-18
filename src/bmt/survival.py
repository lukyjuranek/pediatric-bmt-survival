"""Kaplan–Meier, log-rank tests, and Cox models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError
from scipy.linalg import LinAlgWarning
from lifelines.statistics import (
    multivariate_logrank_test,
    pairwise_logrank_test,
    proportional_hazard_test,
)
from sklearn.model_selection import KFold

from bmt.data import SCREEN_BINARY, SCREEN_NUMERIC, SCREEN_ORDINAL
from bmt.features import COX_BINARY, COX_COLUMNS, impute_numeric

RANDOM_STATE = 42
TIME = "survival_time"
EVENT = "survival_status"


def fit_km(time: pd.Series, event: pd.Series, label: str | None = None) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event, label=label)
    return kmf


def reverse_km_median_followup(time: pd.Series, event: pd.Series) -> float:
    """Median follow-up by treating censoring as the event (reverse KM)."""
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=(event == 0).astype(int))
    return float(kmf.median_survival_time_)


def logrank(time: pd.Series, event: pd.Series, groups: pd.Series):
    return multivariate_logrank_test(time, groups, event_observed=event)


def pairwise_logrank(time: pd.Series, event: pd.Series, groups: pd.Series) -> pd.DataFrame:
    result = pairwise_logrank_test(time, groups, event_observed=event)
    summary = result.summary.copy()
    n_pairs = len(summary)
    summary["p_bonferroni"] = (summary["p"] * n_pairs).clip(upper=1.0)
    return summary.reset_index()


def _cox_fitter(penalizer: float, l1_ratio: float) -> CoxPHFitter:
    return CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)


def fit_cox(
    frame: pd.DataFrame,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
    strata: list[str] | None = None,
) -> tuple[CoxPHFitter, float]:
    """Fit a Cox model. On non-convergence, retry with a small ridge penalty.

    Returns the fitter and the penalizer actually used.
    """
    used = float(penalizer)
    strata = strata or None

    def _fit(pen: float) -> CoxPHFitter:
        model = _cox_fitter(pen, l1_ratio)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=LinAlgWarning)
            model.fit(frame, duration_col=TIME, event_col=EVENT, strata=strata)
        return model

    try:
        model = _fit(used)
    except (ConvergenceError, LinAlgWarning):
        used = max(used, 0.01)
        model = _cox_fitter(used, l1_ratio)
        model.fit(frame, duration_col=TIME, event_col=EVENT, strata=strata)
    return model, used


def survival_design(covariates: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    design = covariates.copy()
    design[TIME] = df[TIME].astype(float)
    design[EVENT] = df[EVENT].astype(int)
    return design


def ph_test(model: CoxPHFitter, frame: pd.DataFrame) -> pd.DataFrame:
    result = proportional_hazard_test(model, frame, time_transform="rank")
    summary = result.summary.copy()
    summary = summary.reset_index()
    if "index" in summary.columns and "covariate" not in summary.columns:
        summary = summary.rename(columns={"index": "covariate"})
    return summary


def violating_covariates(ph_summary: pd.DataFrame, columns: list[str], alpha: float = 0.05) -> list[str]:
    if "covariate" not in ph_summary.columns or "p" not in ph_summary.columns:
        return []
    bad = []
    for name in columns:
        part = ph_summary.loc[ph_summary["covariate"] == name, "p"]
        if len(part) and float(part.min()) < alpha:
            bad.append(name)
    return bad


def train_cindex(model: CoxPHFitter) -> float:
    return float(model.concordance_index_)


def cv_cindex(
    covariates: pd.DataFrame,
    df: pd.DataFrame,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
    strata: list[str] | None = None,
    n_splits: int = 5,
) -> dict:
    """5-fold C-index. The imputer is fit inside each training fold."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    strata = strata or []
    for train_idx, test_idx in splitter.split(covariates):
        train_x = covariates.iloc[train_idx]
        test_x = covariates.iloc[test_idx]
        train_x, test_x = impute_numeric(train_x, test_x)
        train_df = survival_design(train_x, df.iloc[train_idx])
        test_df = survival_design(test_x, df.iloc[test_idx])
        model, _ = fit_cox(train_df, penalizer=penalizer, l1_ratio=l1_ratio, strata=strata)
        scores.append(float(model.score(test_df, scoring_method="concordance_index")))
    arr = np.asarray(scores, dtype=float)
    return {"folds": scores, "mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def likelihood_ratio_test(model_a: CoxPHFitter, model_b: CoxPHFitter, df_extra: int) -> dict:
    from scipy.stats import chi2

    stat = float(2 * (model_b.log_likelihood_ - model_a.log_likelihood_))
    stat = max(stat, 0.0)
    p_value = float(chi2.sf(stat, df_extra))
    return {"lr_stat": stat, "df": int(df_extra), "p": p_value}


def cluster_dummies(labels: np.ndarray, reference: int | None = None) -> tuple[pd.DataFrame, int]:
    series = pd.Series(labels, name="cluster")
    if reference is None:
        reference = int(series.value_counts().idxmax())
    dummies = pd.get_dummies(series, prefix="cluster", dtype=int)
    drop_col = f"cluster_{reference}"
    if drop_col not in dummies.columns:
        raise KeyError(f"Reference cluster column {drop_col} missing")
    return dummies.drop(columns=[drop_col]), int(reference)


def mean_survival_by_cluster(
    model: CoxPHFitter,
    covariates: pd.DataFrame,
    labels: np.ndarray,
    times: np.ndarray,
) -> dict[int, pd.Series]:
    """Average Model A survival curves within each cluster."""
    curves = model.predict_survival_function(covariates, times=times)
    labels = pd.Series(labels, index=covariates.index)
    out: dict[int, pd.Series] = {}
    for cluster, index in labels.groupby(labels).groups.items():
        out[int(cluster)] = curves.loc[:, index].mean(axis=1)
    return out


def univariate_screen(df: pd.DataFrame) -> pd.DataFrame:
    """One Cox model per baseline feature. Disease is dummy-coded against ALL."""
    rows: list[dict] = []
    columns = [(c, "numeric") for c in SCREEN_NUMERIC]
    columns += [(c, "binary") for c in SCREEN_BINARY]
    columns += [(c, "ordinal") for c in SCREEN_ORDINAL]
    for column, kind in columns:
        rows.append(_univariate_row(df, column, kind))
    rows.extend(_disease_rows(df))
    table = pd.DataFrame(rows)
    table["abs_log_hr"] = table["log_hr"].abs()
    return table.sort_values(["p", "feature"], ascending=[True, True]).reset_index(drop=True)


def _univariate_row(df: pd.DataFrame, column: str, kind: str) -> dict:
    sub = df[[column, TIME, EVENT]].dropna()
    base = {
        "feature": column,
        "kind": kind,
        "n": int(len(sub)),
        "events": int(sub[EVENT].sum()) if len(sub) else 0,
        "log_hr": np.nan,
        "hr": np.nan,
        "hr_low": np.nan,
        "hr_high": np.nan,
        "p": np.nan,
        "cindex": np.nan,
        "logrank_p": np.nan,
        "penalizer": 0.0,
        "note": "",
    }
    if sub[column].nunique() < 2 or sub[EVENT].sum() < 1:
        base["note"] = "skipped: no variation or no events"
        return base
    design = sub.rename(columns={column: column})
    try:
        model, used = fit_cox(design)
    except Exception as exc:  # noqa: BLE001 — record and continue the screen
        base["note"] = f"fit failed: {exc}"
        return base
    summary = model.summary.loc[column]
    base.update(
        {
            "log_hr": float(summary["coef"]),
            "hr": float(summary["exp(coef)"]),
            "hr_low": float(summary["exp(coef) lower 95%"]),
            "hr_high": float(summary["exp(coef) upper 95%"]),
            "p": float(summary["p"]),
            "cindex": float(model.concordance_index_),
            "penalizer": used,
        }
    )
    if kind in {"binary", "ordinal"} or sub[column].nunique() <= 8:
        try:
            base["logrank_p"] = float(logrank(sub[TIME], sub[EVENT], sub[column]).p_value)
        except Exception:
            base["logrank_p"] = np.nan
    return base


def _disease_rows(df: pd.DataFrame) -> list[dict]:
    sub = df[["Disease", TIME, EVENT]].dropna()
    logrank_p = float(logrank(sub[TIME], sub[EVENT], sub["Disease"]).p_value)
    dummies = pd.get_dummies(sub["Disease"], dtype=int)
    if "ALL" not in dummies.columns:
        return []
    design = dummies.drop(columns=["ALL"])
    design[TIME] = sub[TIME].values
    design[EVENT] = sub[EVENT].values
    try:
        model, used = fit_cox(design)
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "feature": "Disease",
                "kind": "nominal",
                "n": int(len(sub)),
                "events": int(sub[EVENT].sum()),
                "log_hr": np.nan,
                "hr": np.nan,
                "hr_low": np.nan,
                "hr_high": np.nan,
                "p": logrank_p,
                "cindex": np.nan,
                "logrank_p": logrank_p,
                "penalizer": 0.0,
                "note": f"joint log-rank only; Cox failed: {exc}",
            }
        ]
    rows = []
    for level in design.columns.drop([TIME, EVENT]):
        summary = model.summary.loc[level]
        rows.append(
            {
                "feature": f"Disease[{level} vs ALL]",
                "kind": "nominal",
                "n": int(len(sub)),
                "events": int(sub[EVENT].sum()),
                "log_hr": float(summary["coef"]),
                "hr": float(summary["exp(coef)"]),
                "hr_low": float(summary["exp(coef) lower 95%"]),
                "hr_high": float(summary["exp(coef) upper 95%"]),
                "p": float(summary["p"]),
                "cindex": float(model.concordance_index_),
                "logrank_p": logrank_p,
                "penalizer": used,
                "note": "joint Disease model, reference ALL; logrank_p is the multi-group test",
            }
        )
    return rows


def cox_summary_frame(model: CoxPHFitter) -> pd.DataFrame:
    summary = model.summary.copy().reset_index().rename(columns={"index": "covariate", "covariate": "covariate"})
    if summary.columns[0] != "covariate":
        summary = summary.rename(columns={summary.columns[0]: "covariate"})
    return summary


def sign_comparison(model_a: CoxPHFitter, model_c: CoxPHFitter) -> pd.DataFrame:
    """Compare coefficient signs between unpenalized A and ridge C where names overlap in spirit."""
    signs_c = np.sign(model_c.params_)
    rows = []
    mapping = {
        "Recipientage_per5": "Recipientage",
        "Donorage_per10": "Donorage",
        "log_CD34": "log_CD34",
        "Riskgroup": "Riskgroup",
        "Txpostrelapse": "Txpostrelapse",
        "Stemcellsource": "Stemcellsource",
        "Diseasegroup": "Disease_nonmalignant",
    }
    for left, right in mapping.items():
        if left not in model_a.params_.index or right not in signs_c.index:
            continue
        sign_a = float(np.sign(model_a.params_[left]))
        sign_c = float(signs_c[right])
        if left == "Diseasegroup":
            # Model C uses the nonmalignant dummy; malignant HR has the opposite sign.
            sign_c = -sign_c
        rows.append(
            {
                "model_a": left,
                "model_c": right,
                "sign_a": sign_a,
                "sign_c_aligned": sign_c,
                "agree": bool(sign_a == sign_c or sign_a == 0 or sign_c == 0),
            }
        )
    return pd.DataFrame(rows)


def prepare_model_b_frame(
    cox: pd.DataFrame,
    df: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    dummies, reference = cluster_dummies(labels)
    covariates = pd.concat([cox.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    design = survival_design(covariates, df.reset_index(drop=True))
    return covariates, design, reference
