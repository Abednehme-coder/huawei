#!/usr/bin/env python3
"""
Extract per-window flow-level statistical features from a PCAP, for use as
clustering input in scripts/cluster_validate_labels.py.

One CSV row per time bucket (bid = floor(ts / window_sec)), same bucketing as
scripts.pcap_to_images.run_time_window_mode. Computes n_packets, n_bytes,
TCP flag ratios, unique src/dst IP counts + Shannon entropy, packet-size and
inter-arrival-time mean/std. Also carries `in_schedule` and `heuristic_label`
(the existing schedule+SYN-count rule) for side-by-side comparison only --
these two columns are not meant to be used as clustering input features.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.classify_pcap_syn import (  # noqa: E402
    label_window_with_schedule_and_syn,
    schedule_overlaps_utc_window,
)
from scripts.pcap_to_images import (  # noqa: E402
    iter_pcap_packets_with_timestamp,
    parse_ipv4_header,
)

FEATURE_FIELDS = [
    "n_packets",
    "n_bytes",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "syn_ratio",
    "ack_ratio",
    "fin_ratio",
    "rst_ratio",
    "unique_src_ips",
    "src_ip_entropy",
    "unique_dst_ips",
    "dst_ip_entropy",
    "pkt_size_mean",
    "pkt_size_std",
    "iat_mean",
    "iat_std",
]

CSV_FIELDS = (
    ["pcap", "bucket_id", "t0_unix", "t1_unix"]
    + FEATURE_FIELDS
    + ["in_schedule", "heuristic_label"]
)

_TCP_FLAG_SYN = 0x02
_TCP_FLAG_RST = 0x04
_TCP_FLAG_ACK = 0x10
_TCP_FLAG_FIN = 0x01


def shannon_entropy(counts: dict) -> float:
    """Shannon entropy (base 2, bits) of a discrete count distribution."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


class WindowAccumulator:
    """Incremental (single-pass) per-window feature accumulator."""

    def __init__(self) -> None:
        self.n_packets = 0
        self.n_bytes = 0
        self.syn = 0
        self.ack = 0
        self.fin = 0
        self.rst = 0
        self.src_ip_counts: dict[str, int] = defaultdict(int)
        self.dst_ip_counts: dict[str, int] = defaultdict(int)
        self._size_count = 0
        self._size_mean = 0.0
        self._size_m2 = 0.0
        self._last_ts: Optional[float] = None
        self._iat_count = 0
        self._iat_mean = 0.0
        self._iat_m2 = 0.0

    @staticmethod
    def _welford_update(count: int, mean: float, m2: float, x: float) -> tuple[int, float, float]:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
        return count, mean, m2

    def add_packet(self, ts: float, pkt_bytes: bytes) -> None:
        self.n_packets += 1
        size = len(pkt_bytes)
        self.n_bytes += size
        self._size_count, self._size_mean, self._size_m2 = self._welford_update(
            self._size_count, self._size_mean, self._size_m2, float(size)
        )

        if self._last_ts is not None:
            iat = ts - self._last_ts
            self._iat_count, self._iat_mean, self._iat_m2 = self._welford_update(
                self._iat_count, self._iat_mean, self._iat_m2, iat
            )
        self._last_ts = ts

        parsed = parse_ipv4_header(pkt_bytes)
        if parsed is None:
            return
        self.src_ip_counts[parsed.src_ip] += 1
        self.dst_ip_counts[parsed.dst_ip] += 1
        if parsed.proto == 6:  # TCP
            flags = parsed.tcp_flags
            if flags & _TCP_FLAG_SYN:
                self.syn += 1
            if flags & _TCP_FLAG_ACK:
                self.ack += 1
            if flags & _TCP_FLAG_FIN:
                self.fin += 1
            if flags & _TCP_FLAG_RST:
                self.rst += 1

    @property
    def pkt_size_mean(self) -> float:
        return self._size_mean if self._size_count else 0.0

    @property
    def pkt_size_std(self) -> float:
        return math.sqrt(self._size_m2 / self._size_count) if self._size_count else 0.0

    @property
    def iat_mean(self) -> float:
        return self._iat_mean if self._iat_count else 0.0

    @property
    def iat_std(self) -> float:
        return math.sqrt(self._iat_m2 / self._iat_count) if self._iat_count else 0.0

    def to_row(self) -> dict:
        n = self.n_packets
        return {
            "n_packets": n,
            "n_bytes": self.n_bytes,
            "syn_count": self.syn,
            "ack_count": self.ack,
            "fin_count": self.fin,
            "rst_count": self.rst,
            "syn_ratio": self.syn / n if n else 0.0,
            "ack_ratio": self.ack / n if n else 0.0,
            "fin_ratio": self.fin / n if n else 0.0,
            "rst_ratio": self.rst / n if n else 0.0,
            "unique_src_ips": len(self.src_ip_counts),
            "src_ip_entropy": shannon_entropy(self.src_ip_counts),
            "unique_dst_ips": len(self.dst_ip_counts),
            "dst_ip_entropy": shannon_entropy(self.dst_ip_counts),
            "pkt_size_mean": self.pkt_size_mean,
            "pkt_size_std": self.pkt_size_std,
            "iat_mean": self.iat_mean,
            "iat_std": self.iat_std,
        }


def extract_window_features(
    pcap_path: Path,
    window_sec: float,
    cic_schedule_cfg: dict,
    syn_threshold: int,
) -> list[dict]:
    """Single streaming pass over the PCAP; returns one dict per bucket, sorted by bucket_id."""
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")

    buckets: dict[int, WindowAccumulator] = {}
    for ts, pkt in iter_pcap_packets_with_timestamp(pcap_path):
        bid = int(math.floor(ts / window_sec))
        acc = buckets.get(bid)
        if acc is None:
            acc = WindowAccumulator()
            buckets[bid] = acc
        acc.add_packet(ts, pkt)

    rows = []
    for bid in sorted(buckets):
        acc = buckets[bid]
        t0_unix = bid * window_sec
        t1_unix = (bid + 1) * window_sec
        t0_utc = datetime.fromtimestamp(t0_unix, tz=timezone.utc)
        t1_utc = datetime.fromtimestamp(t1_unix, tz=timezone.utc)

        in_sched = schedule_overlaps_utc_window(cic_schedule_cfg, t0_utc, t1_utc, str(pcap_path))
        heuristic_label = label_window_with_schedule_and_syn(
            cic_schedule_cfg, t0_utc, t1_utc, str(pcap_path), acc.syn, syn_threshold
        )

        row = {
            "pcap": str(pcap_path),
            "bucket_id": bid,
            "t0_unix": t0_unix,
            "t1_unix": t1_unix,
            **acc.to_row(),
            "in_schedule": int(in_sched),
            "heuristic_label": heuristic_label,
        }
        rows.append(row)
    return rows


def write_rows_csv(rows: list[dict], out_path: Path, append: bool) -> None:
    file_exists = out_path.exists() and out_path.stat().st_size > 0
    write_header = not (append and file_exists)
    mode = "a" if append else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-window flow-level statistical features from a PCAP to CSV."
    )
    parser.add_argument("--pcap", required=True, type=Path, help="Path to PCAP file.")
    parser.add_argument("--window-sec", type=float, default=0.3, help="Window size in seconds.")
    parser.add_argument(
        "--cic-schedule",
        required=True,
        type=Path,
        help="Path to cic_ddos2019_syn_windows.json.",
    )
    parser.add_argument(
        "--syn-threshold",
        type=int,
        default=100,
        help="SYN count threshold for the (comparison-only) heuristic_label column.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of overwriting (no header rewritten if file already has content).",
    )
    args = parser.parse_args()

    if not args.pcap.is_file():
        raise SystemExit(f"PCAP not found: {args.pcap}")
    if not args.cic_schedule.is_file():
        raise SystemExit(f"CIC schedule file not found: {args.cic_schedule}")

    with open(args.cic_schedule, encoding="utf-8") as f:
        cfg = json.load(f)

    rows = extract_window_features(args.pcap, args.window_sec, cfg, args.syn_threshold)
    write_rows_csv(rows, args.out, args.append)
    print(f"Wrote {len(rows)} window feature rows to {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
