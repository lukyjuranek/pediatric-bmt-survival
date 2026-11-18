"""End-to-end Lab II analysis: EDA, clustering, Kaplan–Meier, and Cox."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="DataFrame Index is not unique, defaulting to incrementing index instead.",
)

from bmt.clustering import fit_pca, fit_pca_2d, profile_clusters, run_dbscan, select_k
from bmt.data import (
    OUTCOMES,
    POST_TRANSPLANT,
    REDUNDANT_GROUPS,
    SCREEN_BINARY,
    SCREEN_NUMERIC,
    SCREEN_ORDINAL,
    any_missing_mask,
    clean,
    load_arff,
    missingness_table,
    sentinel_table,
    validate,
)
from bmt.features import (
    CLUSTER_BINARY,
    CLUSTER_NUMERIC,
    CLUSTER_ORDINAL,
    COX_BINARY,
    COX_COLUMNS,
    build_cox_frame,
    fit_cluster_matrix,
    impute_numeric,
    pearson_with_time,
    spearman_matrix,
    vif_table,
)
from bmt.plots import plot_fig1, plot_fig2, plot_fig3, plot_fig4, plot_missingness, plot_schoenfeld, plot_spearman
from bmt.survival import (
    TIME,
    EVENT,
    cox_summary_frame,
    cv_cindex,
    fit_cox,
    fit_km,
    likelihood_ratio_test,
    logrank,
    mean_survival_by_cluster,
    pairwise_logrank,
    ph_test,
    prepare_model_b_frame,
    reverse_km_median_followup,
    sign_comparison,
    survival_design,
    train_cindex,
    univariate_screen,
    violating_covariates,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_analysis(data_path: str | Path, root: str | Path) -> dict:
    root = Path(root)
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    raw = load_arff(data_path)
    validate(raw)
    df = clean(raw)

    missing = missingness_table(raw)
    sentinels = sentinel_table(raw)
    _write_csv(missing, tables / "missingness.csv")
    _write_csv(sentinels, tables / "sentinels.csv")

    km = fit_km(df[TIME], df[EVENT], label="All patients")
    followup = reverse_km_median_followup(df[TIME], df[EVENT])
    missing_flag = any_missing_mask(raw).astype(int)
    missing_lr = logrank(df[TIME], df[EVENT], missing_flag.reset_index(drop=True))
    baseline_cols = [c for c in raw.columns if c not in POST_TRANSPLANT + OUTCOMES]
    baseline_missing = raw[baseline_cols].isna().any(axis=1).astype(int)
    baseline_lr = logrank(df[TIME], df[EVENT], baseline_missing.reset_index(drop=True))

    univariate = univariate_screen(df)
    _write_csv(univariate, tables / "univariate_cox.csv")
    pearson_cols = SCREEN_NUMERIC + SCREEN_BINARY + SCREEN_ORDINAL
    pearson = pearson_with_time(df, pearson_cols)
    contrast = pearson.merge(
        univariate[["feature", "log_hr", "p", "hr"]],
        on="feature",
        how="left",
    )
    contrast["abs_pearson"] = contrast["pearson_r"].abs()
    contrast["abs_log_hr"] = contrast["log_hr"].abs()
    contrast["rank_pearson"] = contrast["abs_pearson"].rank(ascending=False, method="min")
    contrast["rank_cox"] = contrast["abs_log_hr"].rank(ascending=False, method="min")
    contrast["rank_gap"] = contrast["rank_pearson"] - contrast["rank_cox"]
    _write_csv(contrast.sort_values("rank_gap", key=lambda s: s.abs(), ascending=False), tables / "pearson_vs_cox.csv")

    spearman_cols = CLUSTER_NUMERIC + CLUSTER_BINARY + CLUSTER_ORDINAL
    spearman = spearman_matrix(df, spearman_cols)
    spearman.to_csv(tables / "spearman_cluster_features.csv")
    wide_cols = [c for cols in REDUNDANT_GROUPS.values() for c in cols if c != "Disease"]
    spearman_wide = spearman_matrix(df, wide_cols)
    spearman_wide.to_csv(tables / "spearman_redundant_blocks.csv")

    cox = build_cox_frame(df)
    cox_imputed, _ = impute_numeric(cox, cox.copy())
    vifs = vif_table(cox_imputed)
    _write_csv(vifs, tables / "vif.csv")

    scaled, _pipe = fit_cluster_matrix(df)
    scores, pca = fit_pca(scaled, variance=0.85)
    scores_2d, var_2d = fit_pca_2d(scaled)
    chosen_k, labels, k_table = select_k(scores)
    _write_csv(k_table, tables / "k_selection.csv")
    dbscan = run_dbscan(scores, labels)
    profile = profile_clusters(df, labels)
    _write_csv(profile, tables / "cluster_profile.csv")
    names = {int(r.cluster): r.name for r in profile.itertuples(index=False)}
    assignments = pd.DataFrame({"patient": df.index, "cluster": labels, "name": [names[int(v)] for v in labels]})
    _write_csv(assignments, tables / "cluster_assignments.csv")

    group_lr = logrank(df[TIME], df[EVENT], pd.Series(labels))
    pairwise = None
    if chosen_k > 2:
        pairwise = pairwise_logrank(df[TIME], df[EVENT], pd.Series(labels))
        _write_csv(pairwise, tables / "pairwise_logrank.csv")

    design_a = survival_design(cox_imputed, df)
    model_a, pen_a = fit_cox(design_a, penalizer=0.0)
    ph = ph_test(model_a, design_a)
    _write_csv(ph, tables / "ph_test.csv")
    strata = violating_covariates(ph, COX_BINARY, alpha=0.05)
    model_a_unstrat = model_a
    pen_unstrat = pen_a
    if strata:
        model_a, pen_a = fit_cox(design_a, penalizer=0.0, strata=strata)

    cov_b, design_b, reference = prepare_model_b_frame(cox_imputed, df, labels)
    if strata:
        for column in strata:
            if column not in design_b.columns:
                design_b[column] = design_a[column].values
    model_b, pen_b = fit_cox(design_b, penalizer=0.0, strata=strata or None)
    lrt = likelihood_ratio_test(model_a, model_b, df_extra=int(cov_b.shape[1] - cox_imputed.shape[1]))

    design_c = survival_design(scaled.reset_index(drop=True), df)
    model_c, pen_c = fit_cox(design_c, penalizer=0.1, l1_ratio=0.0)
    signs = sign_comparison(model_a_unstrat, model_c)
    _write_csv(signs, tables / "cox_sign_comparison.csv")

    summary_a = cox_summary_frame(model_a_unstrat)
    summary_a_final = cox_summary_frame(model_a)
    summary_b = cox_summary_frame(model_b)
    summary_c = cox_summary_frame(model_c)
    _write_csv(summary_a, tables / "cox_a_summary.csv")
    _write_csv(summary_a_final, tables / "cox_a_final_summary.csv")
    _write_csv(summary_b, tables / "cox_b_summary.csv")
    _write_csv(summary_c, tables / "cox_c_summary.csv")

    # Cross-validated C-index uses the same strata decision as the reported models.
    cv_a = cv_cindex(cox_imputed, df, penalizer=pen_a if pen_a else 0.0, strata=strata or None)
    cv_b = cv_cindex(cov_b, df, penalizer=pen_b if pen_b else 0.0, strata=strata or None)

    times = np.linspace(0, float(df[TIME].max()), 300)
    predict_frame = cox_imputed.copy()
    cox_curves = mean_survival_by_cluster(model_a, predict_frame, labels, times)

    residuals = model_a_unstrat.compute_residuals(design_a, kind="scaled_schoenfeld")

    plot_missingness(raw, list(missing["column"]), figures / "missingness.pdf")
    plot_spearman(spearman, figures / "spearman_cluster.pdf", "Spearman correlation, clustering features")
    plot_spearman(spearman_wide, figures / "spearman_blocks.pdf", "Spearman correlation, redundant blocks")
    plot_schoenfeld(residuals, df[TIME], figures / "schoenfeld.pdf")
    plot_fig1(
        df[TIME],
        df[EVENT],
        scores_2d,
        var_2d,
        labels,
        names,
        k_table,
        chosen_k,
        figures / "fig1_overview.pdf",
    )
    plot_fig2(
        df[TIME],
        df[EVENT],
        labels,
        names,
        cox_curves,
        float(group_lr.p_value),
        figures / "fig2_km_cox.pdf",
    )
    # Figure 3 shows every Model A coefficient. The stratified refit drops the
    # offending covariate, so it cannot be the feature-importance display.
    plot_fig3(summary_a, figures / "fig3_forest.pdf")
    plot_fig4(univariate, figures / "fig4_univariate.pdf")

    profile_tex = profile[
        [
            "cluster",
            "name",
            "n",
            "events",
            "event_rate",
            "median_survival_days",
            "Recipientage_mean",
            "pct_high_risk",
            "pct_malignant",
            "CD34_median",
        ]
    ].copy()
    profile_tex["median_survival_days"] = profile_tex["median_survival_days"].map(
        lambda v: "not reached" if np.isinf(v) else f"{v:.0f}"
    )
    profile_tex.to_latex(tables / "cluster_profile.tex", index=False, float_format="%.2f", escape=True)

    dbscan_public = {k: v for k, v in dbscan.items() if k != "labels"}
    results = {
        "n": int(len(df)),
        "events": int(df[EVENT].sum()),
        "censored": int((df[EVENT] == 0).sum()),
        "median_survival_days": float(km.median_survival_time_),
        "median_followup_days": followup,
        "n_rows_with_missing": int(missing_flag.sum()),
        "n_rows_with_baseline_missing": int(baseline_missing.sum()),
        "missing_vs_complete_logrank_p": float(missing_lr.p_value),
        "baseline_missing_logrank_p": float(baseline_lr.p_value),
        "pca_n_components": int(pca.n_components_),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "pca_2d_variance": [float(v) for v in var_2d],
        "k": chosen_k,
        "silhouette": float(k_table.loc[k_table["k"] == chosen_k, "silhouette"].iloc[0]),
        "cluster_names": {str(k): v for k, v in names.items()},
        "logrank_stat": float(group_lr.test_statistic),
        "logrank_p": float(group_lr.p_value),
        "cox_a_penalizer": pen_a,
        "cox_a_unstratified_penalizer": pen_unstrat,
        "cox_a_strata": strata,
        "cox_a_cindex_train": train_cindex(model_a),
        "cox_a_unstrat_cindex_train": train_cindex(model_a_unstrat),
        "cox_b_penalizer": pen_b,
        "cox_b_cindex_train": train_cindex(model_b),
        "cox_c_penalizer": pen_c,
        "cox_c_cindex_train": train_cindex(model_c),
        "cox_a_cindex_cv": cv_a,
        "cox_b_cindex_cv": cv_b,
        "reference_cluster": reference,
        "lrt": lrt,
        "dbscan": dbscan_public,
        "vif_max": float(vifs["vif"].max()),
        "top_univariate": univariate.head(8)[["feature", "hr", "p", "logrank_p"]].to_dict(orient="records"),
    }
    (tables / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    return {
        "results": results,
        "df": df,
        "missing": missing,
        "sentinels": sentinels,
        "univariate": univariate,
        "contrast": contrast,
        "spearman": spearman,
        "vifs": vifs,
        "k_table": k_table,
        "profile": profile,
        "pairwise": pairwise,
        "summary_a": summary_a,
        "summary_a_final": summary_a_final,
        "summary_b": summary_b,
        "ph": ph,
        "signs": signs,
        "labels": labels,
    }
