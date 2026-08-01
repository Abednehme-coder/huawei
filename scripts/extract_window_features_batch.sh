#!/usr/bin/env bash
# Extract per-window flow-level statistical features (CSV) from CICDDoS2019 zips,
# one PCAP at a time: unzip -> rename extensionless to .pcap -> extract_window_features.py -> delete unzipped.
# Mirrors process_zips_in_batches.sh's unzip/rename/loop structure but is a separate script:
# the proven image-generation batch scripts are never touched by this workstream.
#
# Usage:
#   cd ~/huawei && bash scripts/extract_window_features_batch.sh
#
# Env (optional):
#   PCAP_ZIP_DIR      Directory containing the .zip files (default: ~/datasets/CICDDoS2019/PCAPs)
#   STAGING_DIR       Where to unzip (default: $PCAP_ZIP_DIR/staging)
#   OUT_CSV           Output CSV, appended across all PCAPs (default: ~/datasets/CICDDoS2019/window_features.csv)
#   REPO              Repo root (default: script's parent dir)
#   WINDOW_SEC        Window size in seconds (default: 0.3, matches the primary model's bucketing)
#   SYN_THRESHOLD     SYN count threshold for the comparison-only heuristic_label column (default: 100)
#   CIC_SCHEDULE_CONFIG  Path to JSON schedule (default: $REPO/scripts/cic_ddos2019_syn_windows.json)
#   PYTHON            e.g. $REPO/.venv/bin/python on servers where system python lacks deps

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO="${REPO:-$SCRIPT_DIR/..}"
REPO=$(cd "$REPO" && pwd)
WINDOW_SEC="${WINDOW_SEC:-0.3}"
SYN_THRESHOLD="${SYN_THRESHOLD:-100}"
CIC_SCHEDULE_CONFIG="${CIC_SCHEDULE_CONFIG:-$REPO/scripts/cic_ddos2019_syn_windows.json}"
PCAP_ZIP_DIR="${PCAP_ZIP_DIR:-$HOME/datasets/CICDDoS2019/PCAPs}"
OUT_CSV="${OUT_CSV:-$HOME/datasets/CICDDoS2019/window_features.csv}"
STAGING_DIR="${STAGING_DIR:-$PCAP_ZIP_DIR/staging}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)}"

PCAP_ZIP_DIR=$(cd "$PCAP_ZIP_DIR" && pwd)
mkdir -p "$STAGING_DIR"
STAGING_DIR=$(cd "$STAGING_DIR" && pwd)
mkdir -p "$(dirname "$OUT_CSV")"

ZIPS=(
  "PCAP-01-12_0-0249.zip"
  "PCAP-01-12_0250-0499.zip"
  "PCAP-01-12_0500-0749.zip"
  "PCAP-01-12_0750-0818.zip"
  "PCAP-03-11.zip"
)

rename_to_pcap() {
  local root="$1"
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    [ "${f%.pcap}" = "$f" ] || continue
    mv -- "$f" "$f.pcap"
  done < <(find "$root" -type f)
}

extract_for_all_pcaps() {
  local root="$1"
  local pcap
  while IFS= read -r pcap; do
    [ -f "$pcap" ] || continue
    echo "=== Extracting features: $pcap ==="
    "$PYTHON" "$REPO/scripts/extract_window_features.py" \
      --pcap "$pcap" \
      --window-sec "$WINDOW_SEC" \
      --cic-schedule "$CIC_SCHEDULE_CONFIG" \
      --syn-threshold "$SYN_THRESHOLD" \
      --out "$OUT_CSV" \
      --append
  done < <(find "$root" -type f -name "*.pcap" | sort)
}

process_one_zip() {
  local z="$1"
  local path="$PCAP_ZIP_DIR/$z"
  if [ ! -f "$path" ]; then
    echo "Skip (not found): $path"
    return 0
  fi
  echo ""
  echo "========== ZIP: $z =========="
  rm -rf "${STAGING_DIR:?}"/*
  echo "Unzipping..."
  unzip -o -q "$path" -d "$STAGING_DIR"
  echo "Renaming extensionless files to .pcap..."
  rename_to_pcap "$STAGING_DIR"
  extract_for_all_pcaps "$STAGING_DIR"
  echo "Clearing staging..."
  rm -rf "${STAGING_DIR:?}"/*
  echo "Done: $z"
}

main() {
  echo "Repo: $REPO"
  echo "PCAP zips dir: $PCAP_ZIP_DIR"
  echo "Output CSV: $OUT_CSV"
  echo "Staging: $STAGING_DIR"
  echo "WINDOW_SEC=$WINDOW_SEC SYN_THRESHOLD=$SYN_THRESHOLD CIC_SCHEDULE_CONFIG=$CIC_SCHEDULE_CONFIG"
  for z in "${ZIPS[@]}"; do
    process_one_zip "$z"
  done
  echo ""
  echo "All zips done. Rows in $OUT_CSV: $(($(wc -l < "$OUT_CSV") - 1))"
}

main
