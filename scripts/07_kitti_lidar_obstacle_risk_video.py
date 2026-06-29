from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm


DATA_ROOT = Path("data/kitti/raw")
OUTPUT_DIR = Path("outputs/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# BEV region config
# -----------------------------
X_MIN = 0.0
X_MAX = 50.0

Y_MIN = -18.0
Y_MAX = 18.0

# KITTI Velodyne coordinates:
# x = forward, y = left, z = up
#
# These values remove most road/ground points.
# Tune if needed:
#   more clutter  -> increase Z_OBSTACLE_MIN, e.g. -1.10
#   too sparse    -> decrease Z_OBSTACLE_MIN, e.g. -1.50
Z_OBSTACLE_MIN = -1.30
Z_OBSTACLE_MAX = 1.20

BEV_RES = 0.15


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


def load_calibration():
    cam = read_calib_file(CALIB_CAM_PATH)
    velo = read_calib_file(CALIB_VELO_PATH)

    P2 = cam["P_rect_02"].reshape(3, 4)

    R_rect = np.eye(4, dtype=np.float32)
    R_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)

    T_velo_cam = np.eye(4, dtype=np.float32)
    T_velo_cam[:3, :3] = velo["R"].reshape(3, 3)
    T_velo_cam[:3, 3] = velo["T"].reshape(3)

    return P2, R_rect, T_velo_cam


def load_velodyne(path: Path):
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


def filter_obstacle_points(points):
    """
    Keep likely obstacle points and remove most ground/road points.

    Velodyne:
        x = forward
        y = left
        z = up
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= X_MIN) & (x <= X_MAX) &
        (y >= Y_MIN) & (y <= Y_MAX) &
        (z >= Z_OBSTACLE_MIN) & (z <= Z_OBSTACLE_MAX)
    )

    return points[mask]


def project_lidar_to_image(points_velo, rgb_bgr, P2, R_rect, T_velo_cam):
    h, w = rgb_bgr.shape[:2]

    xyz = points_velo[:, :3]

    front_mask = xyz[:, 0] > 0.5
    xyz = xyz[front_mask]

    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz_h = np.concatenate([xyz, ones], axis=1)

    pts_cam = (T_velo_cam @ xyz_h.T).T
    pts_rect = (R_rect @ pts_cam.T).T

    valid = pts_rect[:, 2] > 0.1
    pts_rect = pts_rect[valid]

    uvw = (P2 @ pts_rect.T).T

    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    depth = pts_rect[:, 2]

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u = u[inside]
    v = v[inside]
    depth = depth[inside]

    d = np.clip(depth, 0, 50)
    d_norm = 1.0 - (d / 50.0)
    d_u8 = (d_norm * 255).astype(np.uint8)

    colors = cv2.applyColorMap(d_u8.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    lidar_img = np.zeros_like(rgb_bgr)

    for px, py, color in zip(u, v, colors):
        lidar_img[py, px] = color

    lidar_img = cv2.dilate(lidar_img, np.ones((3, 3), np.uint8), iterations=1)

    mask = np.any(lidar_img > 0, axis=2)

    overlay = rgb_bgr.copy()
    overlay[mask] = cv2.addWeighted(rgb_bgr[mask], 0.25, lidar_img[mask], 0.75, 0)

    return overlay


def make_obstacle_bev(points):
    """
    Create obstacle-only BEV density map.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x >= X_MIN) & (x <= X_MAX) &
        (y >= Y_MIN) & (y <= Y_MAX) &
        (z >= Z_OBSTACLE_MIN) & (z <= Z_OBSTACLE_MAX)
    )

    x = x[mask]
    y = y[mask]
    z = z[mask]

    bev_h = int((X_MAX - X_MIN) / BEV_RES)
    bev_w = int((Y_MAX - Y_MIN) / BEV_RES)

    row = ((X_MAX - x) / BEV_RES).astype(np.int32)
    # Flip horizontal axis so BEV visually matches RGB:
    # vehicle-left appears on the left side of the BEV image,
    # vehicle-right appears on the right side.
    col = ((Y_MAX - y) / BEV_RES).astype(np.int32)

    row = np.clip(row, 0, bev_h - 1)
    col = np.clip(col, 0, bev_w - 1)

    density = np.zeros((bev_h, bev_w), dtype=np.float32)
    height = np.zeros((bev_h, bev_w), dtype=np.float32)

    np.add.at(density, (row, col), 1.0)

    z_norm = (z - Z_OBSTACLE_MIN) / (Z_OBSTACLE_MAX - Z_OBSTACLE_MIN + 1e-8)
    z_norm = np.clip(z_norm, 0, 1)
    np.maximum.at(height, (row, col), z_norm)

    if density.max() > 0:
        density = np.log1p(density)
        density = density / density.max()

    bev = 0.8 * density + 0.2 * height

    # Light smoothing makes the map more readable.
    bev = cv2.GaussianBlur(bev, (3, 3), 0)

    if bev.max() > 0:
        bev = bev / bev.max()

    return bev.astype(np.float32)


def bev_to_risk(bev):
    h, w = bev.shape

    # Nearer cells are riskier.
    closeness = np.linspace(0.15, 1.0, h).reshape(h, 1)

    # Center driving corridor is riskier.
    cols = np.arange(w)
    center = w / 2.0
    dist_from_center = np.abs(cols - center) / center

    center_weight = 1.0 - 0.75 * dist_from_center
    center_weight = np.clip(center_weight, 0.25, 1.0).reshape(1, w)

    risk = bev * closeness * center_weight

    # Mild blur for visualization.
    risk = cv2.GaussianBlur(risk, (5, 5), 0)

    if risk.max() > 0:
        risk = risk / risk.max()

    return risk.astype(np.float32)


def colorize_map(x, cmap):
    x = np.clip(x, 0, 1)
    x_u8 = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(x_u8, cmap)


def make_panel(rgb_bgr, lidar_overlay, bev_color, risk_color, frame_idx):
    """
    Make a clean 2x2 panel.

    Important:
    - RGB and LiDAR projection keep camera aspect ratio.
    - BEV and risk maps keep their own top-down aspect ratio.
    - BEV/risk are centered inside wider black panels instead of being stretched.
    """
    top_tile_w = 621
    top_tile_h = 188

    bottom_panel_w = 621
    bottom_panel_h = 376

    rgb_tile = cv2.resize(rgb_bgr, (top_tile_w, top_tile_h))
    lidar_tile = cv2.resize(lidar_overlay, (top_tile_w, top_tile_h))

    # Preserve BEV/risk aspect ratio.
    bev_h, bev_w = bev_color.shape[:2]
    scale = min(bottom_panel_w / bev_w, bottom_panel_h / bev_h)
    new_w = int(bev_w * scale)
    new_h = int(bev_h * scale)

    bev_resized = cv2.resize(bev_color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    risk_resized = cv2.resize(risk_color, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    bev_panel = np.zeros((bottom_panel_h, bottom_panel_w, 3), dtype=np.uint8)
    risk_panel = np.zeros((bottom_panel_h, bottom_panel_w, 3), dtype=np.uint8)

    y0 = (bottom_panel_h - new_h) // 2
    x0 = (bottom_panel_w - new_w) // 2

    bev_panel[y0:y0 + new_h, x0:x0 + new_w] = bev_resized
    risk_panel[y0:y0 + new_h, x0:x0 + new_w] = risk_resized

    top = np.concatenate([rgb_tile, lidar_tile], axis=1)
    bottom = np.concatenate([bev_panel, risk_panel], axis=1)
    panel = np.concatenate([top, bottom], axis=0)

    cv2.putText(panel, f"RGB frame {frame_idx:04d}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Obstacle LiDAR projection", (top_tile_w + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Obstacle BEV", (15, top_tile_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Obstacle risk heatmap", (top_tile_w + 15, top_tile_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    # Direction hints for BEV interpretation.
    cv2.putText(panel, "forward", (bottom_panel_w // 2 - 50, top_tile_h + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)

    cv2.putText(panel, "forward", (top_tile_w + bottom_panel_w // 2 - 50, top_tile_h + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    seq_root = DATA_ROOT / args.date / f"{args.date}_drive_{args.drive}_sync"
    image_dir = seq_root / "image_02" / "data"
    velo_dir = seq_root / "velodyne_points" / "data"

    global CALIB_CAM_PATH, CALIB_VELO_PATH
    CALIB_CAM_PATH = DATA_ROOT / args.date / "calib_cam_to_cam.txt"
    CALIB_VELO_PATH = DATA_ROOT / args.date / "calib_velo_to_cam.txt"

    out_path = OUTPUT_DIR / f"phase2c_lidar_obstacle_risk_{args.date}_drive_{args.drive}.mp4"

    image_paths = sorted(image_dir.glob("*.png"))
    velo_paths = sorted(velo_dir.glob("*.bin"))

    print("Sequence:", seq_root)
    print("Images found:", len(image_paths))
    print("Velodyne scans found:", len(velo_paths))
    print("Output:", out_path)

    if not image_paths or not velo_paths:
        raise RuntimeError("Missing KITTI images or Velodyne scans.")

    n = min(len(image_paths), len(velo_paths))
    if args.max_frames is not None:
        n = min(n, args.max_frames)

    P2, R_rect, T_velo_cam = load_calibration()

    panel_w = 621 * 2
    panel_h = 188 + 376

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_w, panel_h),
    )

    for idx in tqdm(range(n)):
        rgb_bgr = cv2.imread(str(image_paths[idx]))
        points = load_velodyne(velo_paths[idx])

        obstacle_points = filter_obstacle_points(points)

        lidar_overlay = project_lidar_to_image(
            obstacle_points,
            rgb_bgr,
            P2,
            R_rect,
            T_velo_cam,
        )

        bev = make_obstacle_bev(obstacle_points)
        risk = bev_to_risk(bev)

        bev_color = colorize_map(bev, cv2.COLORMAP_INFERNO)
        risk_color = colorize_map(risk, cv2.COLORMAP_JET)

        panel = make_panel(rgb_bgr, lidar_overlay, bev_color, risk_color, idx)

        writer.write(panel)

    writer.release()

    print("Done.")
    print("Saved:", out_path)
    print("")
    print("If the BEV is too sparse, lower Z_OBSTACLE_MIN to -1.50.")
    print("If the BEV has too much road clutter, raise Z_OBSTACLE_MIN to -1.10.")


if __name__ == "__main__":
    main()
