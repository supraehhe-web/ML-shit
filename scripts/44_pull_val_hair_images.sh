#!/usr/bin/env bash
# Copy the validation real-hair images and the Arm B / released-C outputs for an id list
# from the Modal volume to a local folder (read-only on the volume).
# usage: 44_pull_val_hair_images.sh IDS_FILE OUT_DIR   (MODAL_PROFILE must be set)
set -u
ids="$1"; out="$2"
mkdir -p "$out/orig" "$out/B" "$out/Creleased"
fail=0
while read -r id; do
  [ -z "$id" ] && continue
  modal volume get dmahr-vit-vol "data/raw/isic_212/images/$id.jpg" "$out/orig/$id.jpg" --force >/dev/null 2>&1 || { echo "FAIL orig $id"; fail=1; }
  modal volume get dmahr-vit-vol "data/processed/B/$id.png" "$out/B/$id.png" --force >/dev/null 2>&1 || { echo "FAIL B $id"; fail=1; }
  modal volume get dmahr-vit-vol "data/processed/Creleased/$id.png" "$out/Creleased/$id.png" --force >/dev/null 2>&1 || { echo "FAIL C $id"; fail=1; }
done < "$ids"
echo "orig=$(ls "$out/orig" | wc -l) B=$(ls "$out/B" | wc -l) Creleased=$(ls "$out/Creleased" | wc -l) fail=$fail"
