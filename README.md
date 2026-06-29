# LiftDepth Risk: Road-Aware Monocular BEV Occupancy and Object Risk

This project builds a lightweight perception pipeline for KITTI raw driving sequences.

The final demo converts a monocular RGB sequence into:

1. DA2 metric depth
2. road / sidewalk segmentation
3. YOLO road-user tracking
4. road-gated object projection into BEV
5. object occupancy grid
6. object-aware risk heatmap
7. static interactive HTML demo

The project is meant as a clear robotics / autonomous-perception portfolio demo, not a production safety system.

---

## Final pipeline

```text
RGB frame
   ├── DA2 metric depth
   ├── SegFormer road/sidewalk segmentation
   └── YOLO tracking: person, car, bus, truck

DA2 depth + KITTI intrinsics + road mask
   └── projected road BEV

YOLO tracks + DA2 depth + road gate
   └── tracked object BEV footprints

object BEV + temporal smoothing
   └── object-aware occupancy grid

occupancy + distance + center corridor + motion
   └── risk heatmap
```

---

## Setup

Create and activate an environment first:

```bash
conda create -n liftdepth python=3.10 -y
conda activate liftdepth
```

Install dependencies and download DA2 checkpoint:

```bash
bash scripts/setup_env.sh
```

---

## Data

This repo expects KITTI raw synced data under:

```text
data/kitti/raw/
```

Example final demo drives used during development:

```text
2011_09_26_drive_0005_sync
2011_09_26_drive_0011_sync
2011_09_26_drive_0020_sync
```

You can use the downloader script if included:

```bash
bash scripts/01_download_kitti_raw_multi.sh
```

---

## Reproduce final demo

For drive `0005`:

```bash
bash scripts/run_final_demo_0005.sh
```

Open:

```bash
xdg-open demo/kitti_2011_09_26_drive_0005/index.html
```

If your browser blocks local files, run a local server:

```bash
cd demo/kitti_2011_09_26_drive_0005
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

---

## Main scripts

```text
scripts/15_da2_occupancy_grid_video.py
    DA2 metric depth cache + dense occupancy grid video.

scripts/16_yolo_tracking_video.py
    YOLO detection/tracking cache.

scripts/18_road_segmentation_cache_video.py
    SegFormer road/sidewalk segmentation cache.

scripts/19_road_gated_object_bev_risk.py
    Final pipeline: road-gated tracked objects, BEV occupancy, temporal smoothing, risk.

scripts/20_export_html_demo_assets.py
    Exports static interactive HTML demo.
```

---

## Outputs

Generated files are written under:

```text
outputs/
demo/
```

These are ignored by git by default.
