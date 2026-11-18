"""Sanity checks for the ARFF loader and the baseline feature lists."""

from pathlib import Path

from bmt.data import (
    EXPECTED_EVENTS,
    EXPECTED_N,
    EXPECTED_P,
    OUTCOMES,
    POST_TRANSPLANT,
    SENTINEL,
    SENTINEL_COLUMNS,
    clean,
    load_arff,
    validate,
)
from bmt.features import (
    CLUSTER_COLUMNS,
    COX_SOURCE_COLUMNS,
    build_cluster_frame,
    build_cox_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "bone-marrow.arff"


def test_shape_events_and_range():
    raw = load_arff(DATA)
    assert raw.shape == (EXPECTED_N, EXPECTED_P)
    validate(raw)
    assert int((raw["survival_status"] == 1).sum()) == EXPECTED_EVENTS
    assert int((raw["survival_status"] == 0).sum()) == EXPECTED_N - EXPECTED_EVENTS
    assert raw["survival_time"].min() >= 6
    assert raw["survival_time"].max() <= 3364


def test_missing_counts():
    raw = load_arff(DATA)
    assert int(raw["extcGvHD"].isna().sum()) == 31
    assert int(raw["CMVstatus"].isna().sum()) == 16
    assert int(raw["RecipientCMV"].isna().sum()) == 14
    assert int(raw.isna().any(axis=1).sum()) == 45
    assert raw["CD3dCD34"].isna().eq(raw["CD3dkgx10d8"].isna()).all()


def test_sentinels_exist_but_do_not_enter_model_matrices():
    raw = load_arff(DATA)
    assert int((raw["ANCrecovery"] == SENTINEL).sum()) == 5
    assert int((raw["PLTrecovery"] == SENTINEL).sum()) == 17
    assert int((raw["time_to_aGvHD_III_IV"] == SENTINEL).sum()) == 145
    df = clean(raw)
    cluster = build_cluster_frame(df)
    cox = build_cox_frame(df)
    for frame in (cluster, cox):
        numeric = frame.select_dtypes(include="number")
        assert int((numeric == SENTINEL).sum().sum()) == 0
        assert set(frame.columns).isdisjoint(SENTINEL_COLUMNS)


def test_feature_lists_disjoint_from_outcomes_and_post_transplant():
    banned = set(POST_TRANSPLANT + OUTCOMES)
    assert set(CLUSTER_COLUMNS).isdisjoint(banned)
    assert set(COX_SOURCE_COLUMNS).isdisjoint(banned)
    df = clean(load_arff(DATA))
    assert "log_CD34" in df.columns
    assert int(df["log_CD3"].isna().sum()) == 5
    assert set(df["Disease"].dropna().unique()) <= {
        "ALL",
        "AML",
        "chronic",
        "nonmalignant",
        "lymphoma",
    }
