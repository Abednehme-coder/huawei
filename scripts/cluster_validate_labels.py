#!/usr/bin/env python3
"""
Cross-validate CIC-schedule-based ddos/normal window labels against
unsupervised clustering on flow-level statistical features (see
scripts/extract_window_features.py), per capture day.

Design: the CIC schedule alone gives `candidate_label` (already trusted,
unchanged from the existing pipeline). Independently, KMeans(k=2) on
standardized flow features gives a cluster assignment per day; the cluster
that overlaps more with `in_schedule` rows is deemed the "attack" cluster.
Where the two agree, keep the label (`validated_label`). Where they
disagree, the window is ambiguous and is dropped from training rather than
forced either way. If a day's silhouette score is below --min-silhouette,
clustering is judged too weak to trust for that day: it falls back to
keeping 100% of schedule labels (`resolution_mode=schedule_priority`)
instead of dropping anything, so a bad clustering result never silently
throws away good schedule-labeled data.

Second gate, independent of silhouette: when the true attack fraction is a
tiny minority of a whole day's traffic (observed on CICDDoS2019's
testing_day, where the SYN window is only ~5 minutes out of a full day
capture), a whole-day k=2 split can still clear the silhouette bar while
carving the day up along unrelated traffic-volume variance rather than
attack behavior -- the cluster picked via schedule overlap ends up with a
*lower* mean syn_ratio than the cluster it wasn't picked over. That's not a
flood cluster, so before trusting `cluster_validated`, the identified
"attack" cluster must also show a distinctly higher mean syn_ratio than the
other cluster; otherwise this falls back to
`resolution_mode=schedule_priority_low_syn_purity` (same behavior as the
silhouette fallback: keep 100% of schedule labels, drop nothing).

GMM and DBSCAN are run as robustness cross-checks only (reported via
Adjusted Rand Index vs. the KMeans assignment); they never affect
`validated_label` or `resolution_mode`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.extract_window_features import FEATURE_FIELDS  # noqa: E402

DEFAULT_MIN_SILHOUETTE = 0.15
_MIN_ROWS_FOR_CLUSTERING = 4


def day_key_from_pcap_path(pcap_path: str) -> str:
    """Extract a dd-mm-yyyy capture-day key from a pcap filename; 'unknown' if absent."""
    m = re.search(r"(\d{2}-\d{2}-\d{4})", Path(pcap_path).name)
    return m.group(1) if m else "unknown"


def load_features(csv_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df["day_key"] = df["pcap"].apply(day_key_from_pcap_path)
    return df


def cluster_day(
    df_day: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
) -> dict:
    """Standardize features and run KMeans(k=2) + GMM/DBSCAN cross-checks for one day's rows."""
    n = len(df_day)
    if n < _MIN_ROWS_FOR_CLUSTERING:
        return {
            "kmeans_labels": np.zeros(n, dtype=int),
            "silhouette": -1.0,
            "gmm_ari": None,
            "dbscan_ari": None,
            "dbscan_n_clusters": None,
        }

    X = df_day[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    try:
        silhouette = float(silhouette_score(X_scaled, kmeans_labels))
    except ValueError:
        silhouette = -1.0

    result = {
        "kmeans_labels": kmeans_labels,
        "silhouette": silhouette,
        "gmm_ari": None,
        "dbscan_ari": None,
        "dbscan_n_clusters": None,
    }

    try:
        gmm = GaussianMixture(n_components=2, random_state=random_state)
        gmm_labels = gmm.fit_predict(X_scaled)
        result["gmm_ari"] = float(adjusted_rand_score(kmeans_labels, gmm_labels))
    except Exception:
        pass

    try:
        dbscan = DBSCAN(eps=1.5, min_samples=max(5, n // 20))
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters = len({lbl for lbl in dbscan_labels if lbl != -1})
        result["dbscan_n_clusters"] = n_clusters
        if n_clusters >= 2:
            result["dbscan_ari"] = float(adjusted_rand_score(kmeans_labels, dbscan_labels))
    except Exception:
        pass

    return result


def identify_attack_cluster(df_day: pd.DataFrame, kmeans_labels: np.ndarray) -> int:
    """Cluster id (0 or 1) whose members overlap more with in_schedule==1 (majority vote)."""
    in_schedule = df_day["in_schedule"].to_numpy()
    mask0 = kmeans_labels == 0
    mask1 = kmeans_labels == 1
    overlap0 = in_schedule[mask0].mean() if mask0.any() else 0.0
    overlap1 = in_schedule[mask1].mean() if mask1.any() else 0.0
    return 0 if overlap0 >= overlap1 else 1


def attack_cluster_has_flood_signature(
    df_day: pd.DataFrame,
    kmeans_labels: np.ndarray,
    attack_cluster_id: int,
    syn_ratio_col: str = "syn_ratio",
) -> bool:
    """
    Sanity check on top of the schedule-overlap-based cluster pick: a genuine
    SYN-flood cluster should show elevated syn_ratio relative to the other
    cluster. If the cluster identified as "attack" via schedule overlap does
    NOT have a strictly higher mean syn_ratio than the other cluster, the
    overlap-based pick is likely spurious -- e.g. driven by unrelated
    traffic-volume variance dominating a whole-day k=2 split when the true
    attack fraction is a tiny minority of the day -- rather than real flood
    structure.
    """
    other_cluster_id = 1 - attack_cluster_id
    attack_mask = kmeans_labels == attack_cluster_id
    other_mask = kmeans_labels == other_cluster_id
    if not attack_mask.any() or not other_mask.any():
        return False
    attack_syn_ratio = df_day.loc[attack_mask, syn_ratio_col].mean()
    other_syn_ratio = df_day.loc[other_mask, syn_ratio_col].mean()
    return attack_syn_ratio > other_syn_ratio


def resolve_labels(
    df_day: pd.DataFrame,
    kmeans_labels: np.ndarray,
    attack_cluster_id: int,
    silhouette: float,
    min_silhouette: float = DEFAULT_MIN_SILHOUETTE,
) -> tuple[pd.Series, str]:
    """
    Returns (validated_label series with None for dropped rows, resolution_mode).
    resolution_mode: 'cluster_validated' (agree=keep, disagree=drop),
    'schedule_priority' (silhouette too low, or too few rows to cluster), or
    'schedule_priority_low_syn_purity' (silhouette cleared the bar, but the
    identified attack cluster isn't actually syn-heavy relative to the other
    cluster). The two schedule_priority* modes behave identically: keep all
    schedule labels, drop nothing.
    """
    candidate_label = np.where(df_day["in_schedule"].to_numpy() == 1, "ddos", "normal")

    if silhouette < min_silhouette:
        validated = pd.Series(candidate_label, index=df_day.index, dtype=object)
        return validated, "schedule_priority"

    if not attack_cluster_has_flood_signature(df_day, kmeans_labels, attack_cluster_id):
        validated = pd.Series(candidate_label, index=df_day.index, dtype=object)
        return validated, "schedule_priority_low_syn_purity"

    cluster_label = np.where(kmeans_labels == attack_cluster_id, "ddos", "normal")
    agree = candidate_label == cluster_label
    validated = pd.Series(
        np.where(agree, candidate_label, None), index=df_day.index, dtype=object
    )
    return validated, "cluster_validated"


def build_validated_label_map(df_day: pd.DataFrame, validated_label: pd.Series) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for idx, label in validated_label.items():
        if label is None:
            continue
        row = df_day.loc[idx]
        key = f"{Path(row['pcap']).stem}::{int(row['bucket_id'])}"
        label_map[key] = label
    return label_map


def write_report_md(day_reports: list[dict], out_path: Path, min_silhouette: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cluster validation report",
        "",
        f"min_silhouette threshold: {min_silhouette}",
        "",
        "| day | n_windows | n_ddos | n_normal | %dropped | silhouette | agreement% | "
        "resolution_mode | gmm_ari | dbscan_ari | dbscan_n_clusters |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in day_reports:
        lines.append(
            f"| {r['day_key']} | {r['n_windows']} | {r['n_ddos_final']} | {r['n_normal_final']} | "
            f"{r['pct_dropped']} | {r['silhouette']} | {r['agreement_pct']} | {r['resolution_mode']} | "
            f"{r['gmm_ari']} | {r['dbscan_ari']} | {r['dbscan_n_clusters']} |"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_validation(
    csv_paths: list[Path],
    out_json: Path,
    out_report_md: Path,
    min_silhouette: float = DEFAULT_MIN_SILHOUETTE,
    random_state: int = 42,
) -> dict:
    df = load_features(csv_paths)
    feature_cols = FEATURE_FIELDS

    label_map: dict[str, str] = {}
    day_reports = []

    for day_key, df_day in df.groupby("day_key"):
        cluster_result = cluster_day(df_day, feature_cols, random_state=random_state)
        kmeans_labels = cluster_result["kmeans_labels"]
        silhouette = cluster_result["silhouette"]
        attack_cluster_id = identify_attack_cluster(df_day, kmeans_labels)
        validated_label, resolution_mode = resolve_labels(
            df_day, kmeans_labels, attack_cluster_id, silhouette, min_silhouette
        )

        label_map.update(build_validated_label_map(df_day, validated_label))

        candidate_label = np.where(df_day["in_schedule"].to_numpy() == 1, "ddos", "normal")
        cluster_label = np.where(kmeans_labels == attack_cluster_id, "ddos", "normal")
        agreement_pct = float((candidate_label == cluster_label).mean() * 100)

        n_windows = len(df_day)
        n_kept = int(validated_label.notna().sum())
        n_dropped = n_windows - n_kept

        day_reports.append(
            {
                "day_key": day_key,
                "n_windows": n_windows,
                "n_ddos_final": int((validated_label == "ddos").sum()),
                "n_normal_final": int((validated_label == "normal").sum()),
                "pct_dropped": round(100.0 * n_dropped / n_windows, 2) if n_windows else 0.0,
                "silhouette": round(silhouette, 4),
                "agreement_pct": round(agreement_pct, 2),
                "resolution_mode": resolution_mode,
                "gmm_ari": cluster_result["gmm_ari"],
                "dbscan_ari": cluster_result["dbscan_ari"],
                "dbscan_n_clusters": cluster_result["dbscan_n_clusters"],
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)

    write_report_md(day_reports, out_report_md, min_silhouette)
    return {"label_map": label_map, "day_reports": day_reports}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster-validate window labels from extract_window_features.py CSV output."
    )
    parser.add_argument(
        "--features", required=True, nargs="+", type=Path, help="One or more feature CSV paths."
    )
    parser.add_argument(
        "--out-json", required=True, type=Path, help="Output validated label map JSON path."
    )
    parser.add_argument(
        "--out-report", required=True, type=Path, help="Output markdown report path."
    )
    parser.add_argument("--min-silhouette", type=float, default=DEFAULT_MIN_SILHOUETTE)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    result = run_validation(
        args.features, args.out_json, args.out_report, args.min_silhouette, args.random_state
    )
    n_days = len({r["day_key"] for r in result["day_reports"]})
    print(f"Validated labels for {len(result['label_map'])} windows across {n_days} day(s).")
    print(f"Wrote {args.out_json} and {args.out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
