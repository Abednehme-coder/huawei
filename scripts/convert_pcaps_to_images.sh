#!/usr/bin/env bash
# Classify each PCAP as ddos or normal (Python, early exit), then convert to 224x224 images.
# Usage: run from repo root (~/huawei). Set env vars or edit below.
#
# Env vars:
#   PCAP_DIR       Directory containing .pcap files (default: current dir)
#   IMG_ROOT       Image output root with train/ddos, train/normal (default: ~/datasets/CICDDoS2019/images)
#   SYN_THRESHOLD  Above this SYN count -> ddos when using syn_count or as fallback (default: 100)
#   MAX_IMAGES     Max images per PCAP (default: 200)
#   MAX_FILES      Max number of PCAPs to process (default: no limit)
#   CLASSIFY_METHOD   syn_count or cic_schedule (if unset, inherits env from process_zips_in_batches.sh; default here: syn_count)
#   CIC_SCHEDULE_CONFIG  Path to scripts/cic_ddos2019_syn_windows.json (default: under REPO/scripts/)
#   PCAP_SAMPLE_STRATEGY  even (default) or sequential — even spreads images across the full PCAP (large ~200MB files)
#   SYN_ONLY_DDOS         If 1, use --syn-only when class is ddos so attack images are built from TCP SYNs only (more consistent look)
#   PACKETS_PER_IMAGE     Concatenate N consecutive eligible packets per PNG (default 10). Use 1 for old single-packet / existing checkpoints.
#                         N packets is NOT a fixed time window — rate varies (especially under flood). Retrain after changing N; align run_pipeline.py for live capture.
#   CONVERSION_MODE       per_pcap (default): one PCAP label, then packet-group images. per_second_window: each image = all packets in TIME_WINDOW_SECONDS (UTC);
#                         label = CIC schedule overlaps bucket AND syn_count>threshold (WINDOW_LABEL_LOGIC=and|or).
#   TIME_WINDOW_SECONDS   Bucket width in seconds (default 1) for per_second_window.
#   WINDOW_LABEL_LOGIC    and (default) or or — see pcap_to_images.py --help
#   LABEL_SOURCE          schedule_syn (default) or external — see pcap_to_images.py --help
#   EXTERNAL_LABELS_JSON  Path to validated-labels JSON (required when LABEL_SOURCE=external)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO="${REPO:-$SCRIPT_DIR/..}"
REPO=$(cd "$REPO" && pwd)
PCAP_DIR="${PCAP_DIR:-.}"
IMG_ROOT="${IMG_ROOT:-$HOME/datasets/CICDDoS2019/images}"
SYN_THRESHOLD="${SYN_THRESHOLD:-100}"
MAX_IMAGES="${MAX_IMAGES:-200}"
MAX_FILES="${MAX_FILES:-999999}"
CLASSIFY_METHOD="${CLASSIFY_METHOD:-syn_count}"
CIC_SCHEDULE_CONFIG="${CIC_SCHEDULE_CONFIG:-$REPO/scripts/cic_ddos2019_syn_windows.json}"
PCAP_SAMPLE_STRATEGY="${PCAP_SAMPLE_STRATEGY:-even}"
SYN_ONLY_DDOS="${SYN_ONLY_DDOS:-1}"
PACKETS_PER_IMAGE="${PACKETS_PER_IMAGE:-10}"
CONVERSION_MODE="${CONVERSION_MODE:-per_pcap}"
TIME_WINDOW_SECONDS="${TIME_WINDOW_SECONDS:-1}"
WINDOW_LABEL_LOGIC="${WINDOW_LABEL_LOGIC:-and}"
LABEL_SOURCE="${LABEL_SOURCE:-schedule_syn}"
EXTERNAL_LABELS_JSON="${EXTERNAL_LABELS_JSON:-}"

CLASSIFIER="$REPO/scripts/classify_pcap_syn.py"
CONVERTER="$REPO/scripts/pcap_to_images.py"
# Use python3 if available (server often has no 'python' symlink)
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)}"

mkdir -p "$IMG_ROOT/train/ddos" "$IMG_ROOT/train/normal"
cd "$PCAP_DIR" || exit 1

n=0
for pcap in *.pcap; do
  [ -f "$pcap" ] || continue
  [ $n -ge "$MAX_FILES" ] && break

  if [[ "$CONVERSION_MODE" == "per_second_window" ]]; then
    label_args=(--label-source "$LABEL_SOURCE")
    if [[ "$LABEL_SOURCE" == "external" ]]; then
      label_args+=(--external-labels "$EXTERNAL_LABELS_JSON")
    fi
    # Per UTC time bucket: one PNG = all packets in [t, t+TIME_WINDOW_SECONDS); label = schedule + SYN count per bucket
    # (default), or from EXTERNAL_LABELS_JSON when LABEL_SOURCE=external.
    "$PYTHON" "$CONVERTER" \
      --pcap "$(pwd)/$pcap" \
      --split-out-root "$IMG_ROOT/train" \
      --time-window-seconds "$TIME_WINDOW_SECONDS" \
      --cic-schedule "$CIC_SCHEDULE_CONFIG" \
      --window-syn-threshold "$SYN_THRESHOLD" \
      --window-label-logic "$WINDOW_LABEL_LOGIC" \
      --max-images "$MAX_IMAGES" \
      --sample-strategy "$PCAP_SAMPLE_STRATEGY" \
      --prefix "${pcap%.pcap}" \
      "${label_args[@]}"
    echo "$pcap -> per_second_window (${TIME_WINDOW_SECONDS}s buckets, label_source=$LABEL_SOURCE)"
  else
    # 1) Classify whole PCAP (ddos vs normal): SYN count or CIC schedule (+ fallback)
    if [[ "$CLASSIFY_METHOD" == "cic_schedule" ]]; then
      class=$("$PYTHON" "$CLASSIFIER" "$(pwd)/$pcap" "$SYN_THRESHOLD" --method cic_schedule --schedule "$CIC_SCHEDULE_CONFIG" 2>/dev/null) || class=normal
    else
      class=$("$PYTHON" "$CLASSIFIER" "$(pwd)/$pcap" "$SYN_THRESHOLD" 2>/dev/null) || class=normal
    fi

    syn_arg=""
    if [[ "$class" == "ddos" && "$SYN_ONLY_DDOS" == "1" ]]; then
      syn_arg="--syn-only"
    fi

    "$PYTHON" "$CONVERTER" \
      --pcap "$(pwd)/$pcap" \
      --out "$IMG_ROOT/train/$class" \
      --prefix "${pcap%.pcap}" \
      --max-images "$MAX_IMAGES" \
      --packets-per-image "$PACKETS_PER_IMAGE" \
      --sample-strategy "$PCAP_SAMPLE_STRATEGY" \
      $syn_arg

    echo "$pcap -> $class"
  fi
  n=$((n + 1))
done

echo "Processed $n files. Images: ddos=$(find "$IMG_ROOT/train/ddos" -name '*.png' 2>/dev/null | wc -l) normal=$(find "$IMG_ROOT/train/normal" -name '*.png' 2>/dev/null | wc -l)"
