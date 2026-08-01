#!/usr/bin/env bash
# Unzip one CICDDoS2019 PCAP zip and build BOTH image representations from
# the cluster-validated labels (dataset/window_labels_validated_v1.json) in a
# single pass per pcap, then clear staging and delete the zip:
#   1. byte-dump images (existing pcap_to_images.py encoding)      -> dataset/images_per_second_window_0p3_validated_v1
#   2. FlowPic-style packet-histogram images (new encoding)        -> dataset/images_flowpic_0p3_validated_v1
# Doing both from the same unzip avoids a second re-download of the raw PCAPs.
#
# Usage: scripts/generate_validated_images_batch.sh <zip-filename-in-dataset/PCAPs_raw>
set -euo pipefail

ZIP_NAME="$1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

Z="dataset/PCAPs_raw/$ZIP_NAME"
STAGING="dataset/PCAPs_staging"
LABELS="dataset/window_labels_validated_v1.json"
CIC_SCHEDULE="scripts/cic_ddos2019_syn_windows.json"
# split_train_val_test.py expects everything under <root>/train/{ddos,normal}
# initially, then moves a portion out into val/ and test/ -- so images are
# generated straight into the train/ subdir of each dataset root.
BYTEDUMP_OUT="dataset/images_per_second_window_0p3_validated_v1/train"
FLOWPIC_OUT="dataset/images_flowpic_0p3_validated_v1/train"

echo "Unzipping $Z ..."
unzip -o -q "$Z" -d "$STAGING"

echo "Renaming extensionless files to .pcap..."
find "$STAGING" -type f | while IFS= read -r f; do
  [ -f "$f" ] || continue
  [ "${f%.pcap}" = "$f" ] || continue
  mv -- "$f" "$f.pcap"
done

find "$STAGING" -type f -name "*.pcap" | sort | while IFS= read -r pcap; do
  stem=$(basename "$pcap" .pcap)

  echo "-> $pcap (bytedump)"
  .venv/bin/python scripts/pcap_to_images.py \
    --pcap "$pcap" \
    --time-window-seconds 0.3 \
    --split-out-root "$BYTEDUMP_OUT" \
    --cic-schedule "$CIC_SCHEDULE" \
    --label-source external \
    --external-labels "$LABELS" \
    --max-images 1000000 \
    --sample-strategy even \
    --prefix "$stem"

  echo "-> $pcap (flowpic)"
  .venv/bin/python scripts/pcap_to_flowpic_images.py \
    --pcap "$pcap" \
    --out "$FLOWPIC_OUT" \
    --prefix "$stem" \
    --window-sec 0.3 \
    --max-images 1000000 \
    --sample-strategy even \
    --external-labels "$LABELS"
done

echo "ZIP DONE: $ZIP_NAME"
rm -rf "${STAGING:?}"/*
rm -f "$Z"
echo "Staging cleared, zip deleted."
