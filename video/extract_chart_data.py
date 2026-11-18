"""Read-only extraction of plot data for the explainer.

Writes video/chart_data.json. Does not touch figures/ or tables/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmt.clustering import fit_pca_2d  # noqa: E402
from bmt.data import clean, load_arff  # noqa: E402
from bmt.features import (  # noqa: E402
    COX_BINARY,
    build_cox_frame,
    fit_cluster_matrix,
    impute_numeric,
)
from bmt.survival import (  # noqa: E402
    EVENT,
    TIME,
    fit_cox,
    fit_km,
    mean_survival_by_cluster,
    ph_test,
    survival_design,
    violating_covariates,
)
from lifelines import KaplanMeierFitter  # noqa: E402

warnings.filterwarnings("ignore")


def _finite(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _km_step(time: pd.Series, event: pd.Series) -> dict:
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event)
    sf = kmf.survival_function_.iloc[:, 0]
    times = sf.index.to_numpy(dtype=float)
    surv = sf.to_numpy(dtype=float)
    xs = [float(times[0])]
    ys = [float(surv[0])]
    for i in range(1, len(times)):
        xs.append(float(times[i]))
        ys.append(float(surv[i - 1]))
        xs.append(float(times[i]))
        ys.append(float(surv[i]))
    horizon = float(time.max())
    if xs[-1] < horizon:
        xs.append(horizon)
        ys.append(ys[-1])
    return {
        "t": [round(v, 2) for v in xs],
        "s": [round(v, 4) for v in ys],
        "median": _finite(kmf.median_survival_time_),
    }


def main() -> None:
    raw = load_arff(ROOT / "bone-marrow.arff")
    df = clean(raw)
    assignments = pd.read_csv(ROOT / "tables" / "cluster_assignments.csv").sort_values("patient")
    labels = assignments["cluster"].to_numpy(dtype=int)
    if len(labels) != len(df):
        raise SystemExit(f"label length {len(labels)} != {len(df)}")

    scaled, _pipe = fit_cluster_matrix(df)
    scores, var_2d = fit_pca_2d(scaled)

    profile = pd.read_csv(ROOT / "tables" / "cluster_profile.csv").sort_values("cluster")
    k_table = pd.read_csv(ROOT / "tables" / "k_selection.csv")
    cox_table = pd.read_csv(ROOT / "tables" / "cox_a_summary.csv")
    published = json.loads((ROOT / "tables" / "results.json").read_text())

    km_all = _km_step(df[TIME], df[EVENT])
    km_by = {}
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        km_by[str(int(cluster))] = _km_step(df.loc[mask, TIME], df.loc[mask, EVENT])

    cox = build_cox_frame(df)
    cox_imputed, _ = impute_numeric(cox, cox.copy())
    design = survival_design(cox_imputed, df)
    model, _pen = fit_cox(design, penalizer=0.0)
    ph = ph_test(model, design)
    strata = violating_covariates(ph, COX_BINARY, alpha=0.05)
    if strata:
        model, _pen = fit_cox(design, penalizer=0.0, strata=strata)
    grid = np.linspace(0, float(df[TIME].max()), 80)
    curves = mean_survival_by_cluster(model, cox_imputed, labels, grid)
    cox_curves = {}
    for cluster, series in curves.items():
        cox_curves[str(int(cluster))] = {
            "t": [round(float(t), 2) for t in series.index],
            "s": [round(float(v), 4) for v in series.to_numpy()],
        }

    labels_map = {
        "Recipientage_per5": "Recipient age (per 5 years)",
        "Donorage_per10": "Donor age (per 10 years)",
        "log_CD34": "log CD34+ dose",
        "Diseasegroup": "Malignant disease",
        "Riskgroup": "High risk",
        "Txpostrelapse": "Second transplant",
        "HLAmismatch": "HLA mismatch",
        "Stemcellsource": "Peripheral blood",
    }
    rows = []
    for record in cox_table.to_dict(orient="records"):
        name = str(record["covariate"])
        rows.append(
            {
                "name": name,
                "label": labels_map.get(name, name),
                "hr": float(record["exp(coef)"]),
                "lo": float(record["exp(coef) lower 95%"]),
                "hi": float(record["exp(coef) upper 95%"]),
                "p": float(record["p"]),
            }
        )
    rows.sort(key=lambda row: row["hr"])

    payload = {
        "summary": {
            "n": published["n"],
            "p": 37,
            "events": published["events"],
            "censored": published["censored"],
            "t_min": 6,
            "t_max": 3364,
            "followup": published["median_followup_days"],
            "n_missing_rows": published["n_rows_with_missing"],
            "n_baseline_missing": published["n_rows_with_baseline_missing"],
            "baseline_missing_p": published["baseline_missing_logrank_p"],
            "pca_components": published["pca_n_components"],
            "pca_variance": published["pca_variance_explained"],
            "pca_2d": [float(v) for v in var_2d],
            "k": published["k"],
            "silhouette": published["silhouette"],
            "logrank_stat": published["logrank_stat"],
            "logrank_p": published["logrank_p"],
            "lrt_stat": published["lrt"]["lr_stat"],
            "lrt_df": published["lrt"]["df"],
            "lrt_p": published["lrt"]["p"],
            "cindex_train": published["cox_a_cindex_train"],
            "cindex_cv_mean": published["cox_a_cindex_cv"]["mean"],
            "cindex_cv_std": published["cox_a_cindex_cv"]["std"],
            "vif_max": published["vif_max"],
            "strata": strata,
            "dbscan_eps": published["dbscan"]["eps"],
            "dbscan_min_samples": published["dbscan"]["min_samples"],
            "dbscan_clusters": published["dbscan"]["n_clusters"],
            "dbscan_noise": published["dbscan"]["n_noise"],
            "dbscan_ari": published["dbscan"]["ari_vs_kmeans"],
            "age_hr": next(row["hr"] for row in rows if row["name"] == "Recipientage_per5"),
            "age_p": next(row["p"] for row in rows if row["name"] == "Recipientage_per5"),
            "stem_hr": next(row["hr"] for row in rows if row["name"] == "Stemcellsource"),
            "stem_p": next(row["p"] for row in rows if row["name"] == "Stemcellsource"),
        },
        "pca": {
            "x": [round(float(v), 4) for v in scores[:, 0]],
            "y": [round(float(v), 4) for v in scores[:, 1]],
            "label": [int(v) for v in labels],
        },
        "k_table": [
            {
                "k": int(row.k),
                "silhouette": float(row.silhouette),
                "min_size": int(row.min_size),
                "eligible": bool(row.eligible),
            }
            for row in k_table.itertuples(index=False)
        ],
        "clusters": [
            {
                "id": int(row.cluster),
                "name": str(row.name),
                "n": int(row.n),
                "events": int(row.events),
                "median": _finite(row.median_survival_days),
            }
            for row in profile.itertuples(index=False)
        ],
        "km_all": km_all,
        "km": km_by,
        "cox_curves": cox_curves,
        "cox": rows,
        "sentinels": [
            {"column": "ANCrecovery", "n": 5},
            {"column": "PLTrecovery", "n": 17},
            {"column": "time_to_aGvHD_III_IV", "n": 145},
        ],
    }
    out = Path(__file__).resolve().parent / "chart_data.json"
    out.write_text(json.dumps(payload))
    print(
        f"wrote {out} pca={scores.shape} strata={strata} "
        f"km_medians={[km_by[str(i)]['median'] for i in range(5)]}"
    )


if __name__ == "__main__":
    main()
