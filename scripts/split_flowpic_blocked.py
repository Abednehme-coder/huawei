#!/usr/bin/env python3
"""
Blocked (leakage-safe) train/val/test split for the flowpic dataset.

Unlike split_train_val_test.py's default (random per-image, stratified only
by class), this splits by whole *source pcap file* ("stem"), never by
individual 0.3s window -- adjacent windows from the same capture are highly
correlated, so a random per-window split risks near-duplicate windows
leaking across train/test and inflating eval metrics.

This dataset's ddos-bearing stems fall into exactly two contiguous blocks,
matching CICDDoS2019's two separate attack-execution days:
  - 01-12-2018: SAT-01-12-2018_0617/0618/0619 (966 ddos windows)
  - 03-11-2018: SAT-03-11-2018_0137..0144      (1567 ddos windows)

Split:
  - test  = the ENTIRE 01-12-2018 attack block, held out whole -- a
            cross-day generalization test (unseen attack execution, not
            just unseen windows from an attack the model saw part of).
  - val   = the trailing two 03-11-2018 stems (0143, 0144) -- for
            early-stopping / threshold calibration.
  - train = the remaining six 03-11-2018 attack stems.
  - normal-only stems (no ddos at all) are shuffled (fixed seed) and
    greedily bucketed to round out val/test normal counts, remainder to
    train.

Usage: scripts/split_flowpic_blocked.py [--images-root PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

STEM_RE = re.compile(r"^(.*?)_b\d+\.png$")

TEST_STEMS = {"SAT-01-12-2018_0617", "SAT-01-12-2018_0618", "SAT-01-12-2018_0619"}
VAL_STEMS = {"SAT-03-11-2018_0143", "SAT-03-11-2018_0144"}
# everything else in the 03-11-2018 attack block goes to train implicitly
# (it's whatever ddos-bearing stem isn't in TEST_STEMS/VAL_STEMS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "images_flowpic_0p3_validated_v1",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-normal-target-frac", type=float, default=0.10)
    ap.add_argument("--test-normal-target-frac", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_root = args.images_root / "train"
    val_root = args.images_root / "val"
    test_root = args.images_root / "test"
    if not train_root.is_dir():
        raise SystemExit(f"Missing {train_root}")

    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"No class folders in {train_root}")

    # stem -> {class -> [paths]}
    stem_files: dict[str, dict[str, list[Path]]] = {}
    stem_counts: dict[str, dict[str, int]] = {}
    for cls in classes:
        for p in (train_root / cls).glob("*.png"):
            m = STEM_RE.match(p.name)
            stem = m.group(1) if m else p.stem
            stem_files.setdefault(stem, {}).setdefault(cls, []).append(p)
            stem_counts.setdefault(stem, {}).setdefault(cls, 0)
            stem_counts[stem][cls] += 1

    all_stems = set(stem_files.keys())
    ddos_stems = {s for s in all_stems if stem_counts[s].get("ddos", 0) > 0}
    normal_only_stems = sorted(all_stems - ddos_stems)

    missing = (TEST_STEMS | VAL_STEMS) - all_stems
    if missing:
        raise SystemExit(f"Expected ddos stems not found on disk: {missing}")

    train_ddos_stems = ddos_stems - TEST_STEMS - VAL_STEMS

    total_normal = sum(stem_counts[s].get("normal", 0) for s in all_stems)
    test_normal_fixed = sum(stem_counts[s].get("normal", 0) for s in TEST_STEMS)
    val_normal_fixed = sum(stem_counts[s].get("normal", 0) for s in VAL_STEMS)

    test_normal_target = int(total_normal * args.test_normal_target_frac)
    val_normal_target = int(total_normal * args.val_normal_target_frac)
    test_normal_need = max(0, test_normal_target - test_normal_fixed)
    val_normal_need = max(0, val_normal_target - val_normal_fixed)

    # Heavy-tailed stem sizes (median stem ~tens of windows, a handful of
    # outlier "first pcap in a zip" stems have thousands) -- fill smallest
    # stems first so many small stems hit the target precisely instead of
    # one huge stem overshooting it in one step. Shuffle within same-size
    # groups (via a random tiebreak key) so the choice of *which* stems of a
    # given size isn't just filename order.
    rng = random.Random(args.seed)
    tiebreak = {s: rng.random() for s in normal_only_stems}
    ascending_normal_only = sorted(
        normal_only_stems, key=lambda s: (stem_counts[s].get("normal", 0), tiebreak[s])
    )

    assignment: dict[str, str] = {}
    for s in TEST_STEMS:
        assignment[s] = "test"
    for s in VAL_STEMS:
        assignment[s] = "val"
    for s in train_ddos_stems:
        assignment[s] = "train"

    acc_test, acc_val = 0, 0
    for s in ascending_normal_only:
        n = stem_counts[s].get("normal", 0)
        if acc_test < test_normal_need:
            assignment[s] = "test"
            acc_test += n
        elif acc_val < val_normal_need:
            assignment[s] = "val"
            acc_val += n
        else:
            assignment[s] = "train"

    # Summarize
    totals = {"train": {c: 0 for c in classes}, "val": {c: 0 for c in classes}, "test": {c: 0 for c in classes}}
    for s, split in assignment.items():
        for cls in classes:
            totals[split][cls] += stem_counts[s].get(cls, 0)

    print("Planned split (by stem, blocked):")
    for split in ("train", "val", "test"):
        line = ", ".join(f"{c}={totals[split][c]}" for c in classes)
        n_stems = sum(1 for s, sp in assignment.items() if sp == split)
        print(f"  {split}: {line}  ({n_stems} stems)")

    if args.dry_run:
        print("\n[dry-run] no files moved")
        return 0

    val_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        (val_root / cls).mkdir(parents=True, exist_ok=True)
        (test_root / cls).mkdir(parents=True, exist_ok=True)

    moved = 0
    for s, split in assignment.items():
        if split == "train":
            continue
        dst_root = val_root if split == "val" else test_root
        for cls, paths in stem_files.get(s, {}).items():
            for p in paths:
                target = dst_root / cls / p.name
                shutil.move(str(p), str(target))
                moved += 1

    print(f"\nMoved {moved} files. train/ now holds the remainder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
