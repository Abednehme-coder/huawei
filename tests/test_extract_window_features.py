"""Unit tests for scripts/extract_window_features.py and pcap_to_images.parse_ipv4_header."""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.extract_window_features import (  # noqa: E402
    extract_window_features,
    shannon_entropy,
)
from scripts.pcap_to_images import parse_ipv4_header  # noqa: E402

_ETH_DST = b"\x11" * 6
_ETH_SRC = b"\x22" * 6


def _ip_to_bytes(ip: str) -> bytes:
    return bytes(int(part) for part in ip.split("."))


def _build_ipv4_tcp_packet(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    flags: int,
    payload: bytes = b"",
    vlan: int | None = None,
) -> bytes:
    if vlan is not None:
        eth_header = (
            _ETH_DST
            + _ETH_SRC
            + struct.pack(">H", 0x8100)
            + struct.pack(">H", vlan)
            + struct.pack(">H", 0x0800)
        )
    else:
        eth_header = _ETH_DST + _ETH_SRC + struct.pack(">H", 0x0800)

    total_len = 20 + 20 + len(payload)
    ip_header = struct.pack(
        ">BBHHHBBH4s4s",
        0x45,  # version(4) + IHL(4, in 32-bit words) = 5 -> 20 bytes
        0,  # tos
        total_len,
        0,  # identification
        0,  # flags+fragment offset
        64,  # ttl
        6,  # proto = TCP
        0,  # checksum (unused by parser)
        _ip_to_bytes(src_ip),
        _ip_to_bytes(dst_ip),
    )
    tcp_header = struct.pack(
        ">HHIIBBHHH",
        src_port,
        dst_port,
        0,  # seq
        0,  # ack
        5 << 4,  # data offset = 5 words, reserved bits 0
        flags,
        8192,  # window
        0,  # checksum (unused by parser)
        0,  # urgent pointer
    )
    return eth_header + ip_header + tcp_header + payload


def _pack_record(ts: float, data: bytes) -> bytes:
    sec = int(ts)
    usec = int(round((ts - sec) * 1_000_000))
    return struct.pack("<IIII", sec, usec, len(data), len(data)) + data


def _build_classic_pcap(packets: list[tuple[float, bytes]]) -> bytes:
    global_header = struct.pack("<I", 0xA1B2C3D4) + b"\x00" * 20
    body = b"".join(_pack_record(ts, data) for ts, data in packets)
    return global_header + body


# --- shannon_entropy -------------------------------------------------------


def test_shannon_entropy_three_equal_counts() -> None:
    counts = {"a": 1, "b": 1, "c": 1}
    assert math.isclose(shannon_entropy(counts), math.log2(3))


def test_shannon_entropy_single_value_is_zero() -> None:
    assert shannon_entropy({"a": 5}) == 0.0


def test_shannon_entropy_empty_is_zero() -> None:
    assert shannon_entropy({}) == 0.0


# --- parse_ipv4_header ------------------------------------------------------


def test_parse_ipv4_header_basic_tcp_syn() -> None:
    pkt = _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.2", 1234, 80, flags=0x02)
    parsed = parse_ipv4_header(pkt)
    assert parsed is not None
    assert parsed.src_ip == "10.0.0.1"
    assert parsed.dst_ip == "10.0.0.2"
    assert parsed.src_port == 1234
    assert parsed.dst_port == 80
    assert parsed.proto == 6
    assert parsed.tcp_flags == 0x02


def test_parse_ipv4_header_vlan_tagged() -> None:
    pkt = _build_ipv4_tcp_packet(
        "192.168.1.5", "192.168.1.10", 5555, 443, flags=0x10, vlan=42
    )
    parsed = parse_ipv4_header(pkt)
    assert parsed is not None
    assert parsed.src_ip == "192.168.1.5"
    assert parsed.dst_ip == "192.168.1.10"
    assert parsed.tcp_flags == 0x10


def test_parse_ipv4_header_non_ipv4_returns_none() -> None:
    # ARP ethertype (0x0806), arbitrary payload
    pkt = _ETH_DST + _ETH_SRC + struct.pack(">H", 0x0806) + b"\x00" * 28
    assert parse_ipv4_header(pkt) is None


def test_parse_ipv4_header_truncated_tcp_returns_none() -> None:
    pkt = _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.2", 1234, 80, flags=0x02)
    truncated = pkt[:-10]  # cut into the TCP header
    assert parse_ipv4_header(truncated) is None


def test_parse_ipv4_header_too_short_returns_none() -> None:
    assert parse_ipv4_header(b"\x00" * 10) is None


# --- extract_window_features -------------------------------------------------


def _cfg_no_schedule() -> dict:
    return {
        "timezone": "UTC",
        "syn_windows": {},
        "dd_mm_yyyy_to_cic_day": {},
    }


def test_extract_window_features_single_window_stats(tmp_path: Path) -> None:
    # 3 packets, distinct src IPs, all within the same 0.3s bucket (bid=0).
    packets = [
        (0.01, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)),  # SYN
        (0.02, _build_ipv4_tcp_packet("10.0.0.2", "10.0.0.9", 2, 80, flags=0x02)),  # SYN
        (0.03, _build_ipv4_tcp_packet("10.0.0.3", "10.0.0.9", 3, 80, flags=0x10)),  # ACK only
    ]
    pcap_path = tmp_path / "test.pcap"
    pcap_path.write_bytes(_build_classic_pcap(packets))

    rows = extract_window_features(pcap_path, window_sec=0.3, cic_schedule_cfg=_cfg_no_schedule(), syn_threshold=100)

    assert len(rows) == 1
    row = rows[0]
    assert row["bucket_id"] == 0
    assert row["n_packets"] == 3
    assert row["syn_count"] == 2
    assert row["ack_count"] == 1
    assert math.isclose(row["syn_ratio"], 2 / 3)
    assert row["unique_src_ips"] == 3
    assert row["unique_dst_ips"] == 1
    assert math.isclose(row["src_ip_entropy"], math.log2(3))
    assert row["dst_ip_entropy"] == 0.0
    assert row["in_schedule"] == 0
    assert row["heuristic_label"] == "normal"


def test_extract_window_features_splits_across_buckets(tmp_path: Path) -> None:
    packets = [
        (0.05, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)),
        (0.50, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)),
    ]
    pcap_path = tmp_path / "test.pcap"
    pcap_path.write_bytes(_build_classic_pcap(packets))

    rows = extract_window_features(pcap_path, window_sec=0.3, cic_schedule_cfg=_cfg_no_schedule(), syn_threshold=100)

    assert [r["bucket_id"] for r in rows] == [0, 1]
    assert rows[0]["n_packets"] == 1
    assert rows[1]["n_packets"] == 1


def test_extract_window_features_iat_mean(tmp_path: Path) -> None:
    packets = [
        (0.00, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)),
        (0.10, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)),
        (0.20, _build_ipv4_tcp_packet("10.0.0.1", "10.0.0.9", 1, 80, flags=0x02)),
    ]
    pcap_path = tmp_path / "test.pcap"
    pcap_path.write_bytes(_build_classic_pcap(packets))

    rows = extract_window_features(pcap_path, window_sec=1.0, cic_schedule_cfg=_cfg_no_schedule(), syn_threshold=100)

    assert len(rows) == 1
    assert math.isclose(rows[0]["iat_mean"], 0.10, abs_tol=1e-9)
