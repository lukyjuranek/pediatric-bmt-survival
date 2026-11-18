"""Baseline feature matrices, imputation, and collinearity checks.

Clustering uses 13 pre-transplant inputs (the plan's explicit list; Disease
is one-hot encoded afterwards). Cox model A uses 8 covariates, about 10 events
per variable given 85 deaths.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from bmt.data import OUTCOMES, POST_TRANSPLANT

CLUSTER_NUMERIC = ["Recipientage", "Donorage", "log_CD34", "log_CD3"]
CLUSTER_BINARY = [
    "Recipientgender",
    "Gendermatch",
    "Stemcellsource",
    "ABOmatch",
    "Riskgroup",
    "Txpostrelapse",
]
CLUSTER_ORDINAL = ["CMVstatus", "HLAmatch"]
CLUSTER_CATEGORICAL = ["Disease"]
CLUSTER_COLUMNS = CLUSTER_NUMERIC + CLUSTER_BINARY + CLUSTER_ORDINAL + CLUSTER_CATEGORICAL

COX_COLUMNS = [
    "Recipientage_per5",
    "Donorage_per10",
    "log_CD34",
    "Diseasegroup",
    "Riskgroup",
    "Txpostrelapse",
    "HLAmismatch",
    "Stemcellsource",
]
COX_BINARY = [
    "Diseasegroup",
    "Riskgroup",
    "Txpostrelapse",
    "HLAmismatch",
    "Stemcellsource",
]
COX_SOURCE_COLUMNS = [
    "Recipientage",
    "Donorage",
    "CD34kgx10d6",
    "Diseasegroup",
    "Riskgroup",
    "Txpostrelapse",
    "HLAmismatch",
    "Stemcellsource",
]

DISEASE_LEVELS = ["ALL", "AML", "chronic", "lymphoma", "nonmalignant"]

COX_LABELS = {
    "Recipientage_per5": "Recipient age (per 5 years)",
    "Donorage_per10": "Donor age (per 10 years)",
    "log_CD34": "log CD34+ dose (per kg)",
    "Diseasegroup": "Malignant disease",
    "Riskgroup": "High risk",
    "Txpostrelapse": "Second transplant after relapse",
    "HLAmismatch": "HLA mismatch (8/10 or 7/10)",
    "Stemcellsource": "Peripheral-blood stem cells",
}


def assert_no_leakage(columns: list[str]) -> None:
    banned = set(POST_TRANSPLANT + OUTCOMES)
    found = sorted(set(columns) & banned)
    if found:
        raise AssertionError(f"Model columns include outcomes or post-transplant fields: {found}")


def assert_no_sentinel(frame: pd.DataFrame) -> None:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return
    hits = int((numeric == 1_000_000).sum().sum())
    if hits:
        raise AssertionError(f"Sentinel 1000000 entered a model matrix ({hits} cells)")


def build_cluster_frame(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage(CLUSTER_COLUMNS)
    frame = df[CLUSTER_COLUMNS].copy()
    assert_no_sentinel(frame)
    return frame


def build_cox_frame(df: pd.DataFrame) -> pd.DataFrame:
    assert_no_leakage(COX_SOURCE_COLUMNS)
    frame = pd.DataFrame(
        {
            "Recipientage_per5": df["Recipientage"] / 5.0,
            "Donorage_per10": df["Donorage"] / 10.0,
            "log_CD34": df["log_CD34"],
            "Diseasegroup": df["Diseasegroup"],
            "Riskgroup": df["Riskgroup"],
            "Txpostrelapse": df["Txpostrelapse"],
            "HLAmismatch": df["HLAmismatch"],
            "Stemcellsource": df["Stemcellsource"],
        },
        index=df.index,
    )
    assert_no_sentinel(frame)
    return frame[COX_COLUMNS]


def clustering_preprocessor() -> ColumnTransformer:
    """Median for continuous doses/ages; most-frequent for codes; one-hot disease."""
    categorical = CLUSTER_BINARY + CLUSTER_ORDINAL
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), CLUSTER_NUMERIC),
            ("cat", SimpleImputer(strategy="most_frequent"), categorical),
            (
                "disease",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                categories=[DISEASE_LEVELS],
                                sparse_output=False,
                                handle_unknown="ignore",
                            ),
                        ),
                    ]
                ),
                CLUSTER_CATEGORICAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def clustering_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", clustering_preprocessor()),
            ("scale", StandardScaler()),
        ]
    )


def fit_cluster_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    """Impute, one-hot, and standardize the clustering matrix (fit on all rows)."""
    frame = build_cluster_frame(df)
    pipe = clustering_pipeline()
    values = pipe.fit_transform(frame)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    scaled = pd.DataFrame(values, index=frame.index, columns=names)
    assert_no_sentinel(scaled)
    return scaled, pipe


def impute_numeric(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute using training-fold statistics only."""
    train_out = train.copy()
    test_out = test.copy()
    for column in train_out.columns:
        if train_out[column].isna().any() or test_out[column].isna().any():
            fill = train_out[column].median()
            if pd.isna(fill):
                fill = 0.0
            train_out[column] = train_out[column].fillna(fill)
            test_out[column] = test_out[column].fillna(fill)
    return train_out, test_out


def spearman_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    return df[present].corr(method="spearman")


def vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame.dropna().astype(float)
    design = add_constant(complete, has_constant="add")
    rows = []
    for i, name in enumerate(design.columns):
        if name == "const":
            continue
        rows.append({"feature": name, "vif": float(variance_inflation_factor(design.values, i))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def pearson_with_time(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Naive correlation with observed time. Ignores censoring; for contrast only."""
    rows = []
    for column in columns:
        pair = df[[column, "survival_time"]].dropna()
        if pair[column].nunique() < 2:
            corr = float("nan")
        else:
            corr = float(pair[column].corr(pair["survival_time"], method="pearson"))
        rows.append({"feature": column, "pearson_r": corr, "n": int(len(pair))})
    return pd.DataFrame(rows)
