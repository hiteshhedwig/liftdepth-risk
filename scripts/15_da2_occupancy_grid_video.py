from pathlib import Path
import argparse
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


# Depth Anything V2 metric-depth import path
DA2_METRIC_ROOT = Path("external/Depth-Anything-V2/metric_depth").resolve()
sys.path.insert(0, str(DA2_METRIC_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


DATA_ROOT = Path("data/kitti/raw")
CHECKPOINT = Path("checkpoints/depth_anything_v2_metric_vkitti_vits.pth")

OUTPUT_VIDEO_DIR = Path("outputs/videos")
OUTPUT_DEPTH_DIR = Path("outputs/depth_cache/da2")
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DEPTH_DIR.mkdir(parents=True, exist_ok=True)


# BEV / occupancy grid region
FORWARD_MIN = 1.0
FORWARD_MAX = 50.0
SIDE_MIN = -18.0
SIDE_MAX = 18.0
BEV_RES = 0.20

POINT_STRIDE = 4
SKY_CROP_RATIO = 0.28


def read_calib_file(path: Path):
    data = {}

    for line in path.read_text().splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        value = value.strip()

        if not value:
            continue

        try:
            data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
        except ValueError:
            pass

    return data


def read_kitti_intrinsics(calib_path: Path):
    cam = read_calib_file(calib_path)
    P = cam["P_rect_02"].reshape(3, 4)

    fx = float(P[0, 0])
    fy = float(P[1, 1])
    cx = float(P[0, 2])
    cy = float(P[1, 2])

    return fx, fy, cx, cy


def load_da2_model(device):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Expected: checkpoints/depth_anything_v2_metric_vkitti_vits.pth"
        )

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        }
    }

    model = DepthAnythingV2(
        **{
            **model_configs["vits"],
            "max_depth": 80,
        }
    )

    state = torch.load(str(CHECKPOINT), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    return model


def colorize_depth(depth_m: np.ndarray, max_depth: float = 80.0):
    depth = depth_m.astype(np.float32)
    depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth = np.clip(depth, 0.0, max_depth)

    inv = 1.0 - depth / max_depth
    inv = np.clip(inv, 0.0, 1.0)

    u8 = (inv * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)


def colorize_map(x: np.ndarray, cmap):
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def depth_to_points(depth_m: np.ndarray, fx, fy, cx, cy, mono_y_max: float, stride: int):
    """
    KITTI rectified camera convention:
        X = right
        Y = down
        Z = forward
    """
    h, w = depth_m.shape

    y_start = int(h * SKY_CROP_RATIO)

    ys, xs = np.meshgrid(
        np.arange(y_start, h, stride),
        np.arange(0, w, stride),
        indexing="ij",
    )

    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # Ground suppression.
    # Y points downward. Larger Y tends to include road/ground.
    mask = (
        (Z >= FORWARD_MIN) &
        (Z <= FORWARD_MAX) &
        (X >= SIDE_MIN) &
        (X <= SIDE_MAX) &
        (Y > -2.0) &
        (Y < mono_y_max)
    )

    return points[mask]


def points_to_density_grid(points: np.ndarray):
    """
    Convert projected 3D points into BEV density counts.
    """
    X = points[:, 0]  # right positive
    Z = points[:, 2]  # forward

    grid_h = int((FORWARD_MAX - FORWARD_MIN) / BEV_RES)
    grid_w = int((SIDE_MAX - SIDE_MIN) / BEV_RES)

    row = ((FORWARD_MAX - Z) / BEV_RES).astype(np.int32)
    col = ((X - SIDE_MIN) / BEV_RES).astype(np.int32)

    row = np.clip(row, 0, grid_h - 1)
    col = np.clip(col, 0, grid_w - 1)

    density = np.zeros((grid_h, grid_w), dtype=np.float32)
    np.add.at(density, (row, col), 1.0)

    return density


def density_to_occupancy_probability(density: np.ndarray, density_scale: float):
    """
    Simple probabilistic occupancy conversion.

    More points in a cell -> higher occupied probability.

    Formula:
        P(occupied) = 1 - exp(-density / density_scale)
    """
    occ = 1.0 - np.exp(-density / max(density_scale, 1e-6))

    occ = cv2.GaussianBlur(occ, (5, 5), 0)

    occ = np.clip(occ, 0.0, 1.0)
    return occ.astype(np.float32)


def occupancy_to_risk(occ: np.ndarray):
    """
    Risk from occupancy probability.

    risk = occupancy × closeness × center-corridor-weight
    """
    h, w = occ.shape

    closeness = np.linspace(0.15, 1.0, h).reshape(h, 1)

    cols = np.arange(w)
    center = w / 2.0
    dist_from_center = np.abs(cols - center) / center

    center_weight = 1.0 - 0.75 * dist_from_center
    center_weight = np.clip(center_weight, 0.25, 1.0).reshape(1, w)

    risk = occ * closeness * center_weight
    risk = cv2.GaussianBlur(risk, (7, 7), 0)

    if risk.max() > 0:
        risk = risk / risk.max()

    return risk.astype(np.float32)


def put_map_in_panel(map_color: np.ndarray, panel_w: int, panel_h: int):
    h, w = map_color.shape[:2]

    scale = min(panel_w / w, panel_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(map_color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    y0 = (panel_h - new_h) // 2
    x0 = (panel_w - new_w) // 2

    panel[y0:y0 + new_h, x0:x0 + new_w] = resized

    return panel


def make_panel(rgb_bgr, depth_color, occ_color, risk_color, frame_idx, date, drive):
    top_tile_w = 621
    top_tile_h = 188

    bottom_panel_w = 621
    bottom_panel_h = 376

    rgb_tile = cv2.resize(rgb_bgr, (top_tile_w, top_tile_h))
    depth_tile = cv2.resize(depth_color, (top_tile_w, top_tile_h))

    occ_panel = put_map_in_panel(occ_color, bottom_panel_w, bottom_panel_h)
    risk_panel = put_map_in_panel(risk_color, bottom_panel_w, bottom_panel_h)

    top = np.concatenate([rgb_tile, depth_tile], axis=1)
    bottom = np.concatenate([occ_panel, risk_panel], axis=1)
    panel = np.concatenate([top, bottom], axis=0)

    cv2.putText(panel, f"RGB frame {frame_idx:04d}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "DA2 metric depth", (top_tile_w + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "DA2 occupancy grid", (15, top_tile_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Risk from occupancy", (top_tile_w + 15, top_tile_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "P(occupied): black=free/unknown, bright=occupied",
                (15, top_tile_h + bottom_panel_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1, cv2.LINE_AA)

    cv2.putText(panel, f"KITTI {date} drive {drive}",
                (top_tile_w + 15, top_tile_h + bottom_panel_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1, cv2.LINE_AA)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--mono-y-max", type=float, default=1.35)
    parser.add_argument("--stride", type=int, default=POINT_STRIDE)
    parser.add_argument("--density-scale", type=float, default=3.0)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    seq_root = DATA_ROOT / args.date / f"{args.date}_drive_{args.drive}_sync"
    image_dir = seq_root / "image_02" / "data"
    calib_path = DATA_ROOT / args.date / "calib_cam_to_cam.txt"

    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise RuntimeError(f"No images found at: {image_dir}")

    n = len(image_paths)
    if args.max_frames is not None:
        n = min(n, args.max_frames)

    cache_dir = OUTPUT_DEPTH_DIR / args.date / f"drive_{args.drive}" / "da2_metric_vkitti_vits"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy = read_kitti_intrinsics(calib_path)

    print("Sequence:", seq_root)
    print("Images:", len(image_paths))
    print("Frames used:", n)
    print("Intrinsics:", fx, fy, cx, cy)
    print("mono_y_max:", args.mono_y_max)
    print("stride:", args.stride)
    print("density_scale:", args.density_scale)
    print("Depth cache:", cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = load_da2_model(device)

    panel_w = 621 * 2
    panel_h = 188 + 376

    out_path = OUTPUT_VIDEO_DIR / (
        f"phase8a_da2_occupancy_grid_{args.date}_drive_{args.drive}"
        f"_y{args.mono_y_max}_ds{args.density_scale}.mp4"
    )

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_w, panel_h),
    )

    with torch.no_grad():
        for idx in tqdm(range(n)):
            rgb_bgr = cv2.imread(str(image_paths[idx]))
            if rgb_bgr is None:
                continue

            depth_cache_path = cache_dir / f"{idx:010d}.npy"

            if depth_cache_path.exists() and not args.overwrite_cache:
                depth_m = np.load(depth_cache_path)
            else:
                depth_m = model.infer_image(rgb_bgr).astype(np.float32)
                np.save(depth_cache_path, depth_m)

            img_h, img_w = rgb_bgr.shape[:2]
            if depth_m.shape[:2] != (img_h, img_w):
                depth_m = cv2.resize(depth_m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            points = depth_to_points(
                depth_m,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                mono_y_max=args.mono_y_max,
                stride=args.stride,
            )

            density = points_to_density_grid(points)
            occ = density_to_occupancy_probability(density, density_scale=args.density_scale)
            risk = occupancy_to_risk(occ)

            depth_color = colorize_depth(depth_m, max_depth=args.max_depth)
            occ_color = colorize_map(occ, cv2.COLORMAP_INFERNO)
            risk_color = colorize_map(risk, cv2.COLORMAP_JET)

            panel = make_panel(
                rgb_bgr,
                depth_color,
                occ_color,
                risk_color,
                idx,
                args.date,
                args.drive,
            )

            writer.write(panel)

    writer.release()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
