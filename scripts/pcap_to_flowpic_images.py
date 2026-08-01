#!/usr/bin/env python3
"""
Convert PCAP time-windows to 3-channel packet-histogram images, adapted from
FlowPic (Shapira & Shavitt, "FlowPic: Encrypted Internet Traffic
Classification is as Easy as Image Recognition", 2019) -- an alternative to
the raw-byte-dump-reshaped-to-a-square images in pcap_to_images.py.

Each window becomes a (size_bins x time_bins x 3) image:
  - axis 0 (rows): packet size, binned 0..max_packet_size
  - axis 1 (cols): packet arrival time within the window, binned 0..window_sec
  - channel 0: all packets (overall volume/burst shape)
  - channel 1: TCP SYN packets only (flood-relevant bursts)
  - channel 2: TCP ACK|FIN|RST packets (completed/aborted connections)

Bin counts are log1p-scaled against a fixed global --count-clip (not each
image's own min/max), so absolute traffic volume stays a real visual signal
instead of a 5-packet quiet window and a 5,000-packet flood looking
identical once independently rescaled.

Requires external validated labels (see scripts/cluster_validate_labels.py)
-- unlike pcap_to_images.py's run_time_window_mode, there is no schedule_syn
fallback path here; this script is only used for the label-validated
comparison run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pcap_to_images import (  # noqa: E402
    iter_pcap_packets_with_timestamp,
    parse_ipv4_header,
    stratified_indices,
)

_TCP_FLAG_SYN = 0x02
_TCP_FLAG_RST = 0x04
_TCP_FLAG_ACK = 0x10
_TCP_FLAG_FIN = 0x01

DEFAULT_SIZE_BINS = 64
DEFAULT_TIME_BINS = 64
DEFAULT_MAX_PACKET_SIZE = 1500
DEFAULT_COUNT_CLIP = 200.0


def build_flowpic_image(
    packets: list[tuple[float, bytes]],
    window_start: float,
    window_sec: float,
    size_bins: int = DEFAULT_SIZE_BINS,
    time_bins: int = DEFAULT_TIME_BINS,
    max_packet_size: int = DEFAULT_MAX_PACKET_SIZE,
    count_clip: float = DEFAULT_COUNT_CLIP,
) -> np.ndarray:
    """Build a (size_bins, time_bins, 3) uint8 packet-histogram image for one window."""
    hist = np.zeros((size_bins, time_bins, 3), dtype=np.float64)

    for ts, pkt in packets:
        size_bin = min(size_bins - 1, int(len(pkt) / max_packet_size * size_bins))
        t_offset = ts - window_start
        time_bin = min(time_bins - 1, max(0, int(t_offset / window_sec * time_bins)))

        hist[size_bin, time_bin, 0] += 1

        parsed = parse_ipv4_header(pkt)
        if parsed is not None and parsed.proto == 6:
            flags = parsed.tcp_flags
            if flags & _TCP_FLAG_SYN:
                hist[size_bin, time_bin, 1] += 1
            if flags & (_TCP_FLAG_ACK | _TCP_FLAG_FIN | _TCP_FLAG_RST):
                hist[size_bin, time_bin, 2] += 1

    scaled = np.log1p(hist) / math.log1p(count_clip) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def run_time_window_flowpic_mode(
    pcap_path: Path,
    split_out_root: Path,
    prefix: str,
    window_sec: float,
    max_images: int,
    sample_strategy: str,
    external_labels: dict[str, str],
    size_bins: int = DEFAULT_SIZE_BINS,
    time_bins: int = DEFAULT_TIME_BINS,
    max_packet_size: int = DEFAULT_MAX_PACKET_SIZE,
    count_clip: float = DEFAULT_COUNT_CLIP,
) -> int:
    """
    One flowpic image per time bucket whose f"{pcap_stem}::{bucket_id}" key is
    present in external_labels; buckets absent from external_labels are
    dropped before sampling (not written as images at all). Writes under
    split_out_root/ddos and split_out_root/normal.
    """
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if external_labels is None:
        raise ValueError("external_labels is required")

    buckets: dict[int, list[tuple[float, bytes]]] = defaultdict(list)
    for ts, pkt in iter_pcap_packets_with_timestamp(pcap_path):
        bid = int(math.floor(ts / window_sec))
        buckets[bid].append((ts, pkt))

    pcap_stem = pcap_path.stem
    bucket_ids = sorted(bid for bid in buckets if f"{pcap_stem}::{bid}" in external_labels)
    if not bucket_ids:
        print(f"Saved 0 flowpic images under {split_out_root}")
        return 1

    n_b = len(bucket_ids)
    want = min(max_images, n_b)
    if sample_strategy == "even":
        pos = stratified_indices(n_b, want)
        chosen_bucket_ids = [bucket_ids[i] for i in pos]
    else:
        chosen_bucket_ids = bucket_ids[:want]

    ddos_dir = split_out_root / "ddos"
    normal_dir = split_out_root / "normal"
    ddos_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for bid in chosen_bucket_ids:
        window_start = bid * window_sec
        img_arr = build_flowpic_image(
            buckets[bid],
            window_start,
            window_sec,
            size_bins,
            time_bins,
            max_packet_size,
            count_clip,
        )
        cls = external_labels[f"{pcap_stem}::{bid}"]
        out_dir = ddos_dir if cls == "ddos" else normal_dir
        fname = f"{prefix}_b{bid}.png"
        Image.fromarray(img_arr, mode="RGB").save(out_dir / fname)
        saved += 1

    print(
        f"Saved {saved} flowpic images ({window_sec}s buckets, {size_bins}x{time_bins}x3) "
        f"under {split_out_root}/{{ddos,normal}}"
    )
    return 0 if saved > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PCAP time-windows to 3-channel FlowPic-style packet-histogram images."
    )
    parser.add_argument("--pcap", required=True, type=Path, help="Path to PCAP file.")
    parser.add_argument("--out", required=True, type=Path, help="Output split root (ddos/normal subdirs).")
    parser.add_argument("--prefix", required=True, help="Filename prefix (e.g. pcap stem).")
    parser.add_argument("--window-sec", type=float, default=0.3)
    parser.add_argument("--max-images", type=int, default=10_000)
    parser.add_argument("--sample-strategy", choices=["even", "first"], default="even")
    parser.add_argument(
        "--external-labels", required=True, type=Path, help="Path to validated labels JSON."
    )
    parser.add_argument("--size-bins", type=int, default=DEFAULT_SIZE_BINS)
    parser.add_argument("--time-bins", type=int, default=DEFAULT_TIME_BINS)
    parser.add_argument("--max-packet-size", type=int, default=DEFAULT_MAX_PACKET_SIZE)
    parser.add_argument("--count-clip", type=float, default=DEFAULT_COUNT_CLIP)
    args = parser.parse_args()

    with open(args.external_labels, encoding="utf-8") as f:
        external_labels = json.load(f)

    return run_time_window_flowpic_mode(
        pcap_path=args.pcap,
        split_out_root=args.out,
        prefix=args.prefix,
        window_sec=args.window_sec,
        max_images=args.max_images,
        sample_strategy=args.sample_strategy,
        external_labels=external_labels,
        size_bins=args.size_bins,
        time_bins=args.time_bins,
        max_packet_size=args.max_packet_size,
        count_clip=args.count_clip,
    )


if __name__ == "__main__":
    raise SystemExit(main())
