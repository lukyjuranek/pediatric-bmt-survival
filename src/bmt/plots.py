"""Figures for the notebook and the two-page report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from matplotlib.lines import Line2D

from bmt.features import COX_LABELS

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig1(
    time: pd.Series,
    event: pd.Series,
    scores_2d: np.ndarray,
    var_2d: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    k_table: pd.DataFrame,
    chosen_k: int,
    path: Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)

    kmf = KaplanMeierFitter()
    kmf.fit(time, event, label="All patients")
    kmf.plot_survival_function(ax=axes[0], ci_show=True, color="#0072B2", linewidth=1.6)
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Survival probability")
    axes[0].set_title("Overall Kaplan–Meier")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False)

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        axes[1].scatter(
            scores_2d[mask, 0],
            scores_2d[mask, 1],
            s=16,
            color=COLORS[int(cluster) % len(COLORS)],
            label=f"C{cluster}",
            alpha=0.9,
            linewidths=0,
        )
    axes[1].set_xlabel(f"PC1 ({100 * var_2d[0]:.0f}% var.)")
    axes[1].set_ylabel(f"PC2 ({100 * var_2d[1]:.0f}% var.)")
    axes[1].set_title("Patients in PCA space")
    axes[1].legend(frameon=False, markerscale=1.2, loc="best")

    axes[2].plot(k_table["k"], k_table["silhouette"], marker="o", color="#0072B2")
    axes[2].axvline(chosen_k, color="#D55E00", linewidth=1, linestyle="--")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Silhouette")
    axes[2].set_title(f"Chosen K = {chosen_k}")
    axes[2].set_xticks(list(k_table["k"]))
    _save(fig, path)


def plot_fig2(
    time: pd.Series,
    event: pd.Series,
    labels: np.ndarray,
    names: dict[int, str],
    cox_curves: dict[int, pd.Series],
    logrank_p: float,
    path: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.55))
    fitters = []
    clusters = sorted(int(c) for c in np.unique(labels))
    for cluster in clusters:
        color = COLORS[cluster % len(COLORS)]
        mask = labels == cluster
        kmf = KaplanMeierFitter()
        kmf.fit(time[mask], event[mask], label=f"C{cluster}")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=1.7)
        fitters.append(kmf)
        curve = cox_curves[cluster]
        ax.plot(curve.index, curve.values, color=color, linestyle="--", linewidth=1.15)
    handles, _labels = ax.get_legend_handles_labels()
    legend_labels = [f"C{c}: {names.get(c, '')} (n={int((labels == c).sum())})" for c in clusters]
    handles.append(Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.15))
    legend_labels.append("Cox mean (dashed)")
    ax.legend(handles, legend_labels, frameon=False, loc="upper right", fontsize=6.5)
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, float(time.max()))
    ax.set_title(f"Kaplan–Meier by cluster (log-rank p = {logrank_p:.2f})")
    add_at_risk_counts(
        *fitters,
        ax=ax,
        rows_to_show=["At risk"],
        xticks=[0, 1000, 2000, 3000],
        ypos=-0.28,
    )
    fig.subplots_adjust(bottom=0.28, right=0.98, top=0.90)
    _save(fig, path)


def plot_fig3(summary: pd.DataFrame, path: Path) -> None:
    apply_style()
    table = summary.copy()
    if "covariate" not in table.columns:
        table = table.rename(columns={table.columns[0]: "covariate"})
    table["label"] = table["covariate"].map(lambda name: COX_LABELS.get(name, str(name)))
    table = table.sort_values("coef")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.4, 3.15))
    ax.axvline(1.0, color="0.5", linewidth=0.8)
    ax.errorbar(
        table["exp(coef)"],
        y,
        xerr=[
            table["exp(coef)"] - table["exp(coef) lower 95%"],
            table["exp(coef) upper 95%"] - table["exp(coef)"],
        ],
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        elinewidth=1.2,
        capsize=2,
        markersize=4.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"])
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_title("Cox model A")
    ax.set_xscale("log")
    fig.tight_layout()
    _save(fig, path)


def plot_fig4(univariate: pd.DataFrame, path: Path) -> None:
    """Forest of the ten smallest univariate Cox p-values."""
    apply_style()
    top = univariate.dropna(subset=["hr", "p", "hr_low", "hr_high"]).nsmallest(10, "p")
    top = top.iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axvline(1.0, color="0.5", linewidth=0.8)
    ax.errorbar(
        top["hr"],
        y,
        xerr=[top["hr"] - top["hr_low"], top["hr_high"] - top["hr"]],
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        elinewidth=1.1,
        capsize=2,
        markersize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"])
    ax.set_xscale("log")
    ax.set_xlabel("Univariate hazard ratio (95% CI)")
    ax.set_title("Ten strongest univariate Cox associations")
    fig.tight_layout()
    _save(fig, path)


def plot_spearman(corr: pd.DataFrame, path: Path, title: str) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.7},
        linewidths=0.2,
    )
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def plot_missingness(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    apply_style()
    present = [c for c in columns if c in df.columns and df[c].isna().any()]
    if not present:
        return
    matrix = df[present].isna().astype(int)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    sns.heatmap(matrix.T, cmap=["#f4f1ea", "#333333"], cbar=False, ax=ax)
    ax.set_xlabel("Patient")
    ax.set_ylabel("")
    ax.set_title("Missing entries (dark)")
    fig.tight_layout()
    _save(fig, path)


def plot_schoenfeld(residuals: pd.DataFrame, time: pd.Series, path: Path) -> None:
    apply_style()
    columns = list(residuals.columns)
    n = len(columns)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.4 * nrows), squeeze=False)
    event_time = time.loc[residuals.index]
    for ax, column in zip(axes.ravel(), columns):
        ax.scatter(event_time, residuals[column], s=10, alpha=0.75, color="#0072B2", linewidths=0)
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.set_title(COX_LABELS.get(column, column), fontsize=8)
        ax.set_xlabel("Days")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Scaled Schoenfeld residuals", y=1.01)
    fig.tight_layout()
    _save(fig, path)
