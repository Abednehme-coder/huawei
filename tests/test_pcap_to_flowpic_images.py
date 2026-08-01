"""Unit tests for scripts/pcap_to_flowpic_images.py (synthetic data, no PCAP/server dependency)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.pcap_to_flowpic_images import (  # noqa: E402
    build_flowpic_image,
    run_time_window_flowpic_mode,
)
from tests.test_extract_window_features import (  # noqa: E402
    _build_classic_pcap,
    _build_ipv4_tcp_packet,
)

# --- build_flowpic_image -----------------------------------------------------


def test_build_flowpic_image_shape_and_dtype() -> None:
    packets = [(0.01, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02))]
    img = build_flowpic_image(packets, window_start=0.0, window_sec=0.3, size_bins=8, time_bins=8)
    assert img.shape == (8, 8, 3)
    assert img.dtype == np.uint8


def test_build_flowpic_image_empty_window_is_all_zero() -> None:
    img = build_flowpic_image([], window_start=0.0, window_sec=0.3, size_bins=8, time_bins=8)
    assert img.sum() == 0


def test_build_flowpic_image_syn_packet_populates_channels_0_and_1_not_2() -> None:
    # A bare SYN packet (54 bytes: 14 eth + 20 ip + 20 tcp, no payload) at t=0 -> size_bin 0, time_bin 0.
    packets = [(0.0, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02))]
    img = build_flowpic_image(
        packets, window_start=0.0, window_sec=0.3, size_bins=4, time_bins=4, count_clip=1.0
    )
    assert img[0, 0, 0] > 0  # channel 0: all packets
    assert img[0, 0, 1] > 0  # channel 1: SYN
    assert img[0, 0, 2] == 0  # channel 2: ACK|FIN|RST -- SYN-only packet must not count here


def test_build_flowpic_image_ack_packet_populates_channels_0_and_2_not_1() -> None:
    packets = [(0.0, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x10))]
    img = build_flowpic_image(
        packets, window_start=0.0, window_sec=0.3, size_bins=4, time_bins=4, count_clip=1.0
    )
    assert img[0, 0, 0] > 0
    assert img[0, 0, 1] == 0
    assert img[0, 0, 2] > 0


def test_build_flowpic_image_time_bin_placement() -> None:
    # window_sec=1.0, time_bins=10 -> each bin covers 0.1s. A packet at t_offset=0.55 -> bin 5.
    packets = [(0.55, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02))]
    img = build_flowpic_image(
        packets, window_start=0.0, window_sec=1.0, size_bins=4, time_bins=10, count_clip=1.0
    )
    assert img[:, 5, :].sum() > 0
    assert img[:, :5, :].sum() == 0
    assert img[:, 6:, :].sum() == 0


def test_build_flowpic_image_count_clip_saturates_at_255() -> None:
    # 500 identical SYN packets at the same (size, time) bin, count_clip=10 -> well past saturation.
    packets = [
        (0.0, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)) for _ in range(500)
    ]
    img = build_flowpic_image(
        packets, window_start=0.0, window_sec=0.3, size_bins=4, time_bins=4, count_clip=10.0
    )
    assert img[0, 0, 0] == 255
    assert img[0, 0, 1] == 255


def test_build_flowpic_image_higher_count_is_never_dimmer() -> None:
    few = [(0.0, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02))] * 2
    many = [(0.0, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02))] * 50
    img_few = build_flowpic_image(few, 0.0, 0.3, size_bins=4, time_bins=4, count_clip=200.0)
    img_many = build_flowpic_image(many, 0.0, 0.3, size_bins=4, time_bins=4, count_clip=200.0)
    assert img_many[0, 0, 0] > img_few[0, 0, 0]


# --- run_time_window_flowpic_mode --------------------------------------------


def _make_test_pcap(tmp_path: Path) -> Path:
    """5 packets, one per 0.3s bucket (bid=0..4), all TCP SYN."""
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


def test_flowpic_mode_only_produces_images_for_labeled_buckets(tmp_path: Path) -> None:
    pcap_path = _make_test_pcap(tmp_path)
    out_root = tmp_path / "out"

    external_labels = {
        "x::0": "ddos",
        "x::2": "normal",
        "x::4": "ddos",
        # buckets 1 and 3 intentionally absent -> must be dropped, not defaulted
    }

    rc = run_time_window_flowpic_mode(
        pcap_path=pcap_path,
        split_out_root=out_root,
        prefix="x",
        window_sec=0.3,
        max_images=10,
        sample_strategy="even",
        external_labels=external_labels,
    )

    assert rc == 0
    ddos_files = sorted((out_root / "ddos").glob("*.png"))
    normal_files = sorted((out_root / "normal").glob("*.png"))
    assert len(ddos_files) == 2
    assert len(normal_files) == 1
    assert len(ddos_files) + len(normal_files) == 3  # buckets 1, 3 dropped entirely

    # Saved PNGs really are 3-channel and match the requested (default) bin resolution.
    from PIL import Image

    img = Image.open(ddos_files[0])
    assert img.mode == "RGB"
    assert img.size == (64, 64)  # (width=time_bins, height=size_bins) default


def test_flowpic_mode_requires_external_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_time_window_flowpic_mode(
            pcap_path=Path("unused.pcap"),
            split_out_root=Path("unused"),
            prefix="x",
            window_sec=0.3,
            max_images=10,
            sample_strategy="even",
            external_labels=None,
        )


def test_flowpic_mode_no_labeled_buckets_returns_nonzero(tmp_path: Path) -> None:
    pcap_path = _make_test_pcap(tmp_path)
    out_root = tmp_path / "out"

    rc = run_time_window_flowpic_mode(
        pcap_path=pcap_path,
        split_out_root=out_root,
        prefix="x",
        window_sec=0.3,
        max_images=10,
        sample_strategy="even",
        external_labels={},  # no buckets match
    )
    assert rc == 1
