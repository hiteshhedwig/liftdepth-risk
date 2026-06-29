# Reproduce

## 1. Create environment

```bash
conda create -n liftdepth python=3.10 -y
conda activate liftdepth
bash scripts/setup_env.sh
```

## 2. Download KITTI raw

```bash
bash scripts/01_download_kitti_raw_multi.sh
```

Expected structure:

```text
data/kitti/raw/2011_09_26/
  calib_cam_to_cam.txt
  calib_velo_to_cam.txt
  2011_09_26_drive_0005_sync/
    image_02/data/*.png
    velodyne_points/data/*.bin
    oxts/data/*.txt
```

## 3. Inspect

```bash
python scripts/02_inspect_kitti.py --date 2011_09_26 --drive 0005
```

## 4. Run final pipeline

```bash
bash scripts/run_final_demo_0005.sh
```

This generates:

```text
outputs/tracks/yolo/
outputs/depth_cache/da2/
outputs/seg_cache/road_seg/
outputs/objects/
outputs/videos/
demo/kitti_2011_09_26_drive_0005/
```

## 5. Open demo

```bash
xdg-open demo/kitti_2011_09_26_drive_0005/index.html
```

Or:

```bash
cd demo/kitti_2011_09_26_drive_0005
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```
