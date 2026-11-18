"""Load and validate the UCI bone-marrow ARFF file.

The header contains a ``$`` line (a broken ``%`` comment) and a non-UTF-8
apostrophe, so ``scipy.io.arff`` is not used.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

SENTINEL = 1_000_000
SENTINEL_COLUMNS = ["ANCrecovery", "PLTrecovery", "time_to_aGvHD_III_IV"]

# Observed only after transplant. Not valid baseline covariates.
POST_TRANSPLANT = [
    "Relapse",
    "IIIV",
    "aGvHDIIIIV",
    "extcGvHD",
    "ANCrecovery",
    "PLTrecovery",
    "time_to_aGvHD_III_IV",
]

# Header codes yes as 0 (opposite of IIIV, where 1 = yes).
INVERTED_YES_IS_ZERO = ["aGvHDIIIIV", "extcGvHD"]

OUTCOMES = ["survival_time", "survival_status"]

EXPECTED_N = 187
EXPECTED_P = 37
EXPECTED_EVENTS = 85

REDUNDANT_GROUPS = {
    "recipient_age": ["Recipientage", "Recipientage10", "Recipientageint"],
    "donor_age": ["Donorage", "Donorage35"],
    "hla": ["HLAmatch", "HLAmismatch", "Antigen", "Alel", "HLAgrI"],
    "abo": ["DonorABO", "RecipientABO", "ABOmatch"],
    "cmv": ["DonorCMV", "RecipientCMV", "CMVstatus"],
    "disease": ["Disease", "Diseasegroup"],
    "cell_dose_and_size": ["CD34kgx10d6", "CD3dkgx10d8", "CD3dCD34", "Rbodymass"],
    "sex": ["Recipientgender", "Gendermatch"],
}

# Pre-transplant fields used for the censoring-aware screen.
SCREEN_NUMERIC = [
    "Recipientage",
    "Donorage",
    "CD34kgx10d6",
    "CD3dkgx10d8",
    "CD3dCD34",
    "Rbodymass",
]
SCREEN_BINARY = [
    "Recipientgender",
    "Stemcellsource",
    "Gendermatch",
    "ABOmatch",
    "Riskgroup",
    "Txpostrelapse",
    "Diseasegroup",
    "HLAmismatch",
    "DonorCMV",
    "RecipientCMV",
    "RecipientRh",
    "Donorage35",
    "Recipientage10",
]
SCREEN_ORDINAL = ["CMVstatus", "HLAmatch", "Recipientageint"]

_ATTR_RE = re.compile(r"^@attribute\s+(\S+)\s+", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_path() -> Path:
    return repo_root() / "bone-marrow.arff"


def load_arff(path: str | Path | None = None) -> pd.DataFrame:
    """Parse ``bone-marrow.arff`` (latin-1). ``?`` becomes missing."""
    path = Path(path) if path is not None else default_data_path()
    text = path.read_text(encoding="latin-1")
    names: list[str] = []
    data_lines: list[str] = []
    in_data = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not in_data:
            if stripped.lower().startswith("@data"):
                in_data = True
                continue
            match = _ATTR_RE.match(stripped)
            if match:
                names.append(match.group(1).strip("'\""))
            continue
        data_lines.append(stripped)
    if not names or not data_lines:
        raise ValueError(f"No attributes or data rows found in {path}")
    frame = pd.read_csv(
        StringIO("\n".join(data_lines)),
        header=None,
        names=names,
        na_values=["?"],
        skipinitialspace=True,
    )
    if frame.shape[1] != len(names):
        raise ValueError(f"Expected {len(names)} fields, got {frame.shape[1]}")
    return frame


def validate(df: pd.DataFrame) -> None:
    if df.shape != (EXPECTED_N, EXPECTED_P):
        raise AssertionError(f"Expected {(EXPECTED_N, EXPECTED_P)}, got {df.shape}")
    if df["survival_time"].isna().any() or df["survival_status"].isna().any():
        raise AssertionError("Outcome columns contain missing values")
    events = int((df["survival_status"] == 1).sum())
    if events != EXPECTED_EVENTS:
        raise AssertionError(f"Expected {EXPECTED_EVENTS} deaths, got {events}")
    time = df["survival_time"]
    if time.min() < 6 or time.max() > 3364:
        raise AssertionError(f"survival_time outside [6, 3364]: {time.min()}–{time.max()}")
    leaked = [c for c in SENTINEL_COLUMNS if c not in df.columns]
    if leaked:
        raise AssertionError(f"Missing sentinel columns: {leaked}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cast codes to numbers and add log cell-dose columns.

    Sentinel ``1000000`` values are left untouched here. Post-transplant
    columns that contain them are excluded from every model matrix.
    """
    out = df.copy().reset_index(drop=True)
    for column in out.columns:
        if column == "Disease":
            out[column] = out[column].astype("string").str.strip()
            out.loc[out[column].isin(["", "<NA>", "nan", "None"]), column] = pd.NA
            out[column] = out[column].astype(object).where(out[column].notna(), np.nan)
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    cd34 = out["CD34kgx10d6"]
    cd3 = out["CD3dkgx10d8"]
    if (cd34.dropna() <= 0).any():
        raise ValueError("CD34 dose must be positive before log transform")
    if (cd3.dropna() <= 0).any():
        raise ValueError("CD3 dose must be positive before log transform")
    out["log_CD34"] = np.log(cd34)
    out["log_CD3"] = np.log(cd3)
    return out


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    table = pd.DataFrame(
        {
            "column": counts.index,
            "n_missing": counts.values.astype(int),
            "pct_missing": (100 * counts.values / len(df)).round(2),
        }
    )
    table = table.loc[table["n_missing"] > 0].sort_values(
        ["n_missing", "column"], ascending=[False, True]
    )
    return table.reset_index(drop=True)


def sentinel_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in SENTINEL_COLUMNS:
        mask = df[column] == SENTINEL
        rows.append(
            {
                "column": column,
                "n_sentinel": int(mask.sum()),
                "n_sentinel_and_death": int(((mask) & (df["survival_status"] == 1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def any_missing_mask(df: pd.DataFrame) -> pd.Series:
    feature_cols = [c for c in df.columns if c not in OUTCOMES and c not in ("log_CD34", "log_CD3")]
    return df[feature_cols].isna().any(axis=1)
