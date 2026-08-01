"""Unit tests for scripts/cluster_validate_labels.py (synthetic data, no PCAP/server dependency)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cluster_validate_labels import (  # noqa: E402
    attack_cluster_has_flood_signature,
    build_validated_label_map,
    cluster_day,
    day_key_from_pcap_path,
    identify_attack_cluster,
    resolve_labels,
)
from scripts.extract_window_features import FEATURE_FIELDS  # noqa: E402


def _make_synthetic_day_df(
    n_per_cluster: int,
    separation: float,
    noise_std: float,
    seed: int,
    pcap_name: str = "x.pcap",
    in_schedule_matches_cluster: bool = True,
) -> pd.DataFrame:
    """
    Build a synthetic per-window feature DataFrame with two groups of rows:
    group 0 centered at 0, group 1 centered at `separation`, across all feature dims.
    When in_schedule_matches_cluster=True, in_schedule==1 exactly for group 1 rows
    (the ddos-labeling case this module is meant to validate). When False,
    in_schedule is assigned independently of group membership (no real structure).
    """
    rng = np.random.default_rng(seed)
    n = n_per_cluster
    feat0 = rng.normal(0.0, noise_std, size=(n, len(FEATURE_FIELDS)))
    feat1 = rng.normal(separation, noise_std, size=(n, len(FEATURE_FIELDS)))
    X = np.vstack([feat0, feat1])

    df = pd.DataFrame(X, columns=FEATURE_FIELDS)
    if in_schedule_matches_cluster:
        df["in_schedule"] = [0] * n + [1] * n
    else:
        df["in_schedule"] = rng.integers(0, 2, size=2 * n)
    df["pcap"] = pcap_name
    df["bucket_id"] = range(2 * n)
    return df


def test_day_key_from_pcap_path_extracts_date() -> None:
    assert day_key_from_pcap_path("/data/SAT-01-12-2018_0750.pcap") == "01-12-2018"


def test_day_key_from_pcap_path_unknown_when_absent() -> None:
    assert day_key_from_pcap_path("nodate.pcap") == "unknown"


def test_separable_blobs_high_silhouette_and_full_agreement() -> None:
    df_day = _make_synthetic_day_df(
        n_per_cluster=30, separation=12.0, noise_std=1.0, seed=0, in_schedule_matches_cluster=True
    )

    result = cluster_day(df_day, FEATURE_FIELDS, random_state=42)
    assert result["silhouette"] > 0.5

    attack_cluster_id = identify_attack_cluster(df_day, result["kmeans_labels"])
    validated_label, resolution_mode = resolve_labels(
        df_day, result["kmeans_labels"], attack_cluster_id, result["silhouette"], min_silhouette=0.15
    )

    assert resolution_mode == "cluster_validated"
    assert validated_label.isna().sum() == 0  # perfectly separable -> nothing dropped

    label_map = build_validated_label_map(df_day, validated_label)
    assert len(label_map) == len(df_day)
    # Group 1 rows (in_schedule==1) must all resolve to "ddos", group 0 to "normal".
    for bucket_id in range(30):
        assert label_map[f"x::{bucket_id}"] == "normal"
    for bucket_id in range(30, 60):
        assert label_map[f"x::{bucket_id}"] == "ddos"


def test_noisy_overlapping_frame_triggers_silhouette_fallback_and_drops_nothing() -> None:
    # No real cluster structure: single unimodal blob, in_schedule uncorrelated with features.
    df_day = _make_synthetic_day_df(
        n_per_cluster=30, separation=0.0, noise_std=5.0, seed=0, in_schedule_matches_cluster=False
    )

    result = cluster_day(df_day, FEATURE_FIELDS, random_state=42)
    assert result["silhouette"] < 0.15

    attack_cluster_id = identify_attack_cluster(df_day, result["kmeans_labels"])
    validated_label, resolution_mode = resolve_labels(
        df_day, result["kmeans_labels"], attack_cluster_id, result["silhouette"], min_silhouette=0.15
    )

    assert resolution_mode == "schedule_priority"
    assert validated_label.isna().sum() == 0  # fallback keeps 100% of schedule labels

    candidate_label = np.where(df_day["in_schedule"].to_numpy() == 1, "ddos", "normal")
    assert (validated_label.to_numpy() == candidate_label).all()


def test_disagreement_rows_are_dropped_not_forced() -> None:
    # Separable clusters, but in_schedule disagrees with cluster membership for some rows.
    df_day = _make_synthetic_day_df(
        n_per_cluster=30, separation=12.0, noise_std=1.0, seed=0, in_schedule_matches_cluster=True
    )
    # Flip in_schedule for 5 rows in group 1 (they'll disagree with their cluster's majority label).
    df_day.loc[30:34, "in_schedule"] = 0

    result = cluster_day(df_day, FEATURE_FIELDS, random_state=42)
    assert result["silhouette"] > 0.5

    attack_cluster_id = identify_attack_cluster(df_day, result["kmeans_labels"])
    validated_label, resolution_mode = resolve_labels(
        df_day, result["kmeans_labels"], attack_cluster_id, result["silhouette"], min_silhouette=0.15
    )

    assert resolution_mode == "cluster_validated"
    assert validated_label.isna().sum() == 5
    label_map = build_validated_label_map(df_day, validated_label)
    for bucket_id in range(30, 35):
        assert f"x::{bucket_id}" not in label_map


def test_attack_cluster_without_flood_signature_falls_back_to_schedule_priority() -> None:
    """
    Reproduces the real failure mode found on CICDDoS2019's testing_day (01-12-2018):
    a whole-day k=2 split can find two well-separated clusters driven by unrelated
    traffic-volume variance, where the cluster with (barely) higher in_schedule
    overlap rate has a LOWER mean syn_ratio than the other cluster -- i.e. it isn't
    actually a flood cluster. Even though silhouette clears the bar and the
    overlap-based selection "works" mechanically, this must not be trusted as
    cluster_validated.
    """
    rng = np.random.default_rng(0)
    n0, n1 = 400, 100
    cols = FEATURE_FIELDS
    syn_idx = cols.index("syn_ratio")

    # Cluster 0: large, offset from cluster 1 on all other dims, but LOW syn_ratio.
    # A tiny number of in_schedule=1 rows land here -- enough to give it a small but
    # nonzero majority overlap rate over cluster 1.
    feat0 = rng.normal(0.0, 0.5, size=(n0, len(cols)))
    feat0[:, syn_idx] = rng.normal(0.1, 0.02, size=n0)

    # Cluster 1: smaller, offset on all other dims, HIGH syn_ratio (the real
    # flood-consistent signature) but zero in_schedule overlap here, mirroring the
    # observed data where schedule-overlap-based selection picked the wrong cluster.
    feat1 = rng.normal(6.0, 0.5, size=(n1, len(cols)))
    feat1[:, syn_idx] = rng.normal(0.8, 0.05, size=n1)

    X = np.vstack([feat0, feat1])
    df_day = pd.DataFrame(X, columns=cols)
    in_schedule = np.zeros(n0 + n1, dtype=int)
    in_schedule[:8] = 1  # 8 of 400 cluster-0 rows in-schedule (~2% overlap rate)
    df_day["in_schedule"] = in_schedule
    df_day["pcap"] = "y.pcap"
    df_day["bucket_id"] = range(n0 + n1)

    result = cluster_day(df_day, FEATURE_FIELDS, random_state=42)
    assert result["silhouette"] > 0.15  # well separated -- by volume, not attack signal

    kmeans_labels = result["kmeans_labels"]
    attack_cluster_id = identify_attack_cluster(df_day, kmeans_labels)
    assert not attack_cluster_has_flood_signature(df_day, kmeans_labels, attack_cluster_id)

    validated_label, resolution_mode = resolve_labels(
        df_day, kmeans_labels, attack_cluster_id, result["silhouette"], min_silhouette=0.15
    )
    assert resolution_mode == "schedule_priority_low_syn_purity"
    assert validated_label.isna().sum() == 0  # fallback keeps 100% of schedule labels

    candidate_label = np.where(df_day["in_schedule"].to_numpy() == 1, "ddos", "normal")
    assert (validated_label.to_numpy() == candidate_label).all()


def test_too_few_rows_falls_back_to_schedule_priority() -> None:
    df_day = _make_synthetic_day_df(
        n_per_cluster=1, separation=12.0, noise_std=1.0, seed=0, in_schedule_matches_cluster=True
    )
    result = cluster_day(df_day, FEATURE_FIELDS, random_state=42)
    assert result["silhouette"] == -1.0

    attack_cluster_id = identify_attack_cluster(df_day, result["kmeans_labels"])
    validated_label, resolution_mode = resolve_labels(
        df_day, result["kmeans_labels"], attack_cluster_id, result["silhouette"], min_silhouette=0.15
    )
    assert resolution_mode == "schedule_priority"
    assert validated_label.isna().sum() == 0
