#!/usr/bin/env bash
set -euo pipefail

# Reproduce the final HTML demo for KITTI raw drive 0005.
#
# Expected data:
#   data/kitti/raw/2011_09_26/2011_09_26_drive_0005_sync
#
# If data is missing, run:
#   bash scripts/01_download_kitti_raw_multi.sh
#
# Then:
#   bash scripts/run_final_demo_0005.sh

DATE="2011_09_26"
DRIVE="0005"
YOLO_MODEL="${YOLO_MODEL:-yolov8m.pt}"

echo "Running final road-gated object-aware BEV/risk pipeline..."
python scripts/19_road_gated_object_bev_risk.py \
  --date "$DATE" \
  --drive "$DRIVE" \
  --yolo-model "$YOLO_MODEL" \
  --dilate-road-px 12 \
  --footprint-scale 0.65 \
  --obj-smooth-alpha 0.70 \
  --grid-smooth-alpha 0.20

echo
echo "Exporting static HTML demo..."
python scripts/20_export_html_demo_assets.py \
  --date "$DATE" \
  --drive "$DRIVE" \
  --overwrite

echo
echo "Done."
echo "Open:"
echo "  xdg-open demo/kitti_${DATE}_drive_${DRIVE}/index.html"
echo
echo "If browser blocks local files:"
echo "  cd demo/kitti_${DATE}_drive_${DRIVE}"
echo "  python -m http.server 8000"
echo "  # open http://localhost:8000"
