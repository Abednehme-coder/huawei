#!/usr/bin/env python3
"""Cross-reference the flagged-stem false positives against the official
CICDDoS2019 published attack schedule and the CICFlowMeter Syn.csv labels.

Official schedule (scripts/cic_ddos2019_syn_windows.json): testing_day
(01-12-2018) SYN window is 13:29-13:34 America/Halifax (AST, UTC-4 on this
date, DST ended 2018-11-04) = 17:29:00-17:34:00 UTC. window_features_real.csv
t0_unix/t1_unix are raw pcap epoch seconds (UTC, no ambiguity -- see
scripts/extract_window_features.py). Syn.csv's `Timestamp` column is in
America/Halifax local time (self-consistent: its own per-flow timestamps
cluster in 13:29:xx-13:34:xx, matching the schedule almost exactly).

Scratch/one-off script for the 2026-08-05 CSV cross-check (see
HANDOVER_2026-08-05.md section 3 / WORK_LOG.md).
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = _ROOT / "dataset" / "CIC_official_labels" / "flagged_windows_predictions.csv"
FEATURES_CSV = _ROOT / "dataset" / "window_features_real.csv"
SYN_CSV = _ROOT / "dataset" / "CIC_official_labels" / "extracted" / "01-12" / "Syn.csv"

OFFICIAL_START_UTC = dt.datetime(2018, 12, 1, 17, 29, 0, tzinfo=dt.timezone.utc)
OFFICIAL_END_UTC = dt.datetime(2018, 12, 1, 17, 34, 0, tzinfo=dt.timezone.utc)
HALIFAX_OFFSET_HOURS = 4  # AST = UTC-4 on 2018-12-01 (standard time, DST ended 2018-11-04)


def load_bucket_times():
    bucket_t0 = {}
    with open(FEATURES_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            if "SAT-01-12-2018_0617" in row["pcap"] or "SAT-01-12-2018_0619" in row["pcap"]:
                bucket_t0[row["bucket_id"]] = float(row["t0_unix"])
    return bucket_t0


def main():
    bucket_t0 = load_bucket_times()
    print(f"Loaded t0_unix for {len(bucket_t0)} buckets in the two flagged stems.")

    fp_rows = []
    with open(PRED_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["true_label"] == "normal" and row["pred_label"] == "ddos":
                t0 = bucket_t0.get(row["bucket_id"])
                if t0 is not None:
                    fp_rows.append((row["bucket_id"], t0))

    print(f"False positives with resolved t0_unix: {len(fp_rows)}")

    in_window = 0
    near_window = 0  # within 60s of the official window on either side
    far = 0
    min_t, max_t = None, None
    for bucket_id, t0 in fp_rows:
        t = dt.datetime.fromtimestamp(t0, tz=dt.timezone.utc)
        if min_t is None or t < min_t:
            min_t = t
        if max_t is None or t > max_t:
            max_t = t
        if OFFICIAL_START_UTC <= t <= OFFICIAL_END_UTC:
            in_window += 1
        elif (OFFICIAL_START_UTC - dt.timedelta(seconds=60)) <= t <= (OFFICIAL_END_UTC + dt.timedelta(seconds=60)):
            near_window += 1
        else:
            far += 1

    print(f"FP time range (UTC): {min_t} .. {max_t}")
    print(f"Official schedule window (UTC): {OFFICIAL_START_UTC} .. {OFFICIAL_END_UTC}")
    print(f"FPs strictly inside official window: {in_window}")
    print(f"FPs within 60s of official window (boundary-adjacent): {near_window}")
    print(f"FPs far from official window (>60s away): {far}")

    # Now cross-check the "far" FPs specifically against Syn.csv per-flow
    # labels, since those are the ones that would NOT be explained by the
    # schedule-boundary theory -- if THEY also show up as Syn-labeled in
    # the official per-flow CSV, that's a stronger, independent confirmation
    # than the schedule table alone.
    print("\nScanning Syn.csv for flows whose (Halifax-local) Timestamp falls")
    print("within +/-2s of each far/near FP's window, to check the official")
    print("per-flow Label independent of the schedule table...")

    # Build a lookup from Halifax-local second -> list of Labels, but only
    # for the relevant hour to keep this cheap (avoid loading 1.58M rows'
    # worth of parsed datetimes if avoidable -- stream once, bucket by
    # integer local ns from 13:00-14:00 only).
    second_labels = {}  # local "HH:MM:SS" (no fractional) -> Counter-ish list
    with open(SYN_CSV) as f:
        r = csv.DictReader(f)
        ts_key = " Timestamp" if " Timestamp" in r.fieldnames else "Timestamp"
        label_key = " Label" if " Label" in r.fieldnames else "Label"
        for row in r:
            ts_s = row.get(ts_key)
            if not ts_s or not ts_s.startswith("2018-12-01 1"):
                continue
            sec = ts_s[:19]  # "YYYY-MM-DD HH:MM:SS"
            second_labels.setdefault(sec, []).append(row.get(label_key))

    print(f"Indexed {sum(len(v) for v in second_labels.values())} Syn.csv flows across {len(second_labels)} distinct seconds.")

    def official_label_near(t0_utc: float, tol_s: int = 2):
        t_utc = dt.datetime.fromtimestamp(t0_utc, tz=dt.timezone.utc)
        t_local = t_utc - dt.timedelta(hours=HALIFAX_OFFSET_HOURS)
        labels = []
        for delta in range(-tol_s, tol_s + 1):
            key = (t_local + dt.timedelta(seconds=delta)).strftime("%Y-%m-%d %H:%M:%S")
            labels.extend(second_labels.get(key, []))
        return labels

    checked = 0
    matched_syn = 0
    matched_benign = 0
    no_flow_found = 0
    for bucket_id, t0 in fp_rows:
        labels = official_label_near(t0)
        checked += 1
        if not labels:
            no_flow_found += 1
        elif any(l == "Syn" for l in labels):
            matched_syn += 1
        else:
            matched_benign += 1

    print(f"\nOf {checked} flagged false-positive windows:")
    print(f"  {matched_syn} have an official CICFlowMeter flow labeled 'Syn' within +/-2s")
    print(f"  {matched_benign} have only 'BENIGN' official flows within +/-2s")
    print(f"  {no_flow_found} have no official flow record within +/-2s at all")


if __name__ == "__main__":
    main()
