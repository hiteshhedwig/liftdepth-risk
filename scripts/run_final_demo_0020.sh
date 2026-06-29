#!/usr/bin/env bash
set -euo pipefail

DATE="2011_09_26"
DRIVE="0020"
YOLO_MODEL="${YOLO_MODEL:-yolov8m.pt}"

python scripts/19_road_gated_object_bev_risk.py \
  --date "$DATE" \
  --drive "$DRIVE" \
  --yolo-model "$YOLO_MODEL" \
  --dilate-road-px 12 \
  --footprint-scale 0.65 \
  --obj-smooth-alpha 0.70 \
  --grid-smooth-alpha 0.20

python scripts/20_export_html_demo_assets.py \
  --date "$DATE" \
  --drive "$DRIVE" \
  --overwrite

echo "Open: demo/kitti_${DATE}_drive_${DRIVE}/index.html"
