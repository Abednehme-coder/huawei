"""
Regression + external-label-path tests for scripts.pcap_to_images.run_time_window_mode.

Verifies:
1. label_source="external" only produces images for buckets present in the
   external-labels JSON (absent buckets are dropped before sampling, not
   just unlabeled), and routes each to the directory the JSON says.
2. The default label_source="schedule_syn" path is unaffected by the new
   flags -- byte-for-byte same behavior as before this change (regression guard).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.pcap_to_images import run_time_window_mode  # noqa: E402
from tests.test_extract_window_features import (  # noqa: E402
    _build_classic_pcap,
    _build_ipv4_tcp_packet,
)

_CIC_SCHEDULE = _REPO / "scripts" / "cic_ddos2019_syn_windows.json"


def _make_test_pcap(tmp_path: Path, window_sec: float = 0.3) -> Path:
    """5 packets, one per bucket (bid=0..4), all TCP SYN, distinct enough for images."""
    packets = [
        (0.05, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)),  # bucket 0
        (0.35, _build_ipv4_tcp_packet("10.0.0.2", "10.0.0.9", 2, 80, flags=0x02)),  # bucket 1
        (0.65, _build_ipv4_tcp_packet("10.0.0.3", "10.0.0.9", 3, 80, flags=0x02)),  # bucket 2
        (0.95, _build_ipv4_tcp_packet("10.0.0.4", "10.0.0.9", 4, 80, flags=0x02)),  # bucket 3
        (1.25, _build_ipv4_tcp_packet("10.0.0.5", "10.0.0.9", 5, 80, flags=0x02)),  # bucket 4
    ]
    pcap_path = tmp_path / "x.pcap"
    pcap_path.write_bytes(_build_classic_pcap(packets))
    return pcap_path


def test_external_labels_only_produces_images_for_labeled_buckets(tmp_path: Path) -> None:
    pcap_path = _make_test_pcap(tmp_path)
    out_root = tmp_path / "out"

    external_labels = {
        "x::0": "ddos",
        "x::2": "normal",
        "x::4": "ddos",
        # buckets 1 and 3 intentionally absent -> must be dropped, not defaulted
    }

    rc = run_time_window_mode(
        pcap_path=pcap_path,
        split_out_root=out_root,
        prefix="x",
        window_sec=0.3,
        max_images=10,
        sample_strategy="even",
        cic_schedule=_CIC_SCHEDULE,
        syn_threshold=100,
        window_logic="and",
        label_source="external",
        external_labels=external_labels,
    )

    assert rc == 0
    ddos_files = sorted((out_root / "ddos").glob("*.png"))
    normal_files = sorted((out_root / "normal").glob("*.png"))
    assert len(ddos_files) == 2
    assert len(normal_files) == 1
    assert len(ddos_files) + len(normal_files) == 3  # buckets 1, 3 dropped entirely


def test_external_labels_requires_dict() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_time_window_mode(
            pcap_path=Path("unused.pcap"),
            split_out_root=Path("unused"),
            prefix="x",
            window_sec=0.3,
            max_images=10,
            sample_strategy="even",
            cic_schedule=_CIC_SCHEDULE,
            syn_threshold=100,
            window_logic="and",
            label_source="external",
            external_labels=None,
        )


def _make_test_pcap_varying_syn_counts(tmp_path: Path) -> Path:
    """5 buckets (0.3s each) with 1..5 SYN packets respectively, so per-bucket syn_count is
    unique even though run_time_window_mode's filename truncates t0_unix to whole seconds
    (a pre-existing characteristic of 0.3s windows, unrelated to this change) -- keeps this
    regression test's file count assertion meaningful."""
    packets = []
    for bid in range(5):
        t0 = bid * 0.3
        n_syn = bid + 1
        for j in range(n_syn):
            ts = t0 + 0.01 * (j + 1)
            packets.append(
                (ts, _build_ipv4_tcp_packet(f"10.0.{bid}.{j}", "10.0.0.9", bid + 1, 80, flags=0x02))
            )
    pcap_path = tmp_path / "x.pcap"
    pcap_path.write_bytes(_build_classic_pcap(packets))
    return pcap_path


def test_default_schedule_syn_path_unaffected_by_new_flags(tmp_path: Path) -> None:
    # Epoch-time (1970) timestamps never fall inside the CIC schedule's real 2018 windows,
    # and syn_count never exceeds the default syn_threshold=100 -> every bucket must resolve
    # to "normal" under the default (unchanged) schedule_syn rule.
    pcap_path = _make_test_pcap_varying_syn_counts(tmp_path)
    out_root = tmp_path / "out"

    rc = run_time_window_mode(
        pcap_path=pcap_path,
        split_out_root=out_root,
        prefix="x",
        window_sec=0.3,
        max_images=10,
        sample_strategy="even",
        cic_schedule=_CIC_SCHEDULE,
        syn_threshold=100,
        window_logic="and",
        # label_source omitted -> must default to "schedule_syn", external_labels omitted -> None
    )

    assert rc == 0
    ddos_files = list((out_root / "ddos").glob("*.png"))
    normal_files = list((out_root / "normal").glob("*.png"))
    assert len(ddos_files) == 0
    assert len(normal_files) == 5
