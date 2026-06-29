from pathlib import Path
import argparse
import json
import subprocess
import sys

import cv2
import numpy as np
from tqdm import tqdm


DATA_ROOT = Path("data/kitti/raw")

TRACK_ROOT = Path("outputs/tracks/yolo")
DEPTH_CACHE_ROOT = Path("outputs/depth_cache/da2")
SEG_CACHE_ROOT = Path("outputs/seg_cache/road_seg")

OUTPUT_VIDEO_DIR = Path("outputs/videos")
OUTPUT_OBJECT_DIR = Path("outputs/objects")

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_OBJECT_DIR.mkdir(parents=True, exist_ok=True)


FORWARD_MIN = 1.0
FORWARD_MAX = 50.0
SIDE_MIN = -18.0
SIDE_MAX = 18.0
BEV_RES = 0.20


# Only these classes.
# COCO: 0 person, 2 car, 5 bus, 7 truck
ALLOWED_CLASS_IDS = {0, 2, 5, 7}
VEHICLE_CLASSES = {"car", "bus", "truck"}
PERSON_CLASSES = {"person"}

CLASS_FOOTPRINTS = {
    "person": (0.8, 0.8),   # width, length
    "car": (2.0, 4.2),
    "bus": (2.8, 9.0),
    "truck": (2.8, 7.0),
}


CLASS_RISK_WEIGHTS = {
    "person": 1.35,
    "car": 1.0,
    "bus": 1.15,
    "truck": 1.15,
}


def track_superclass(class_name: str):
    class_name = str(class_name).lower()
    if class_name in {"car", "bus", "truck"}:
        return "vehicle"
    if class_name == "person":
        return "person"
    return class_name


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter + 1e-6
    return float(inter / denom)


def bbox_center_distance_norm(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)

    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    scale = max((aw * ah) ** 0.5, (bw * bh) ** 0.5, 1.0)
    return float(dist / scale)


class StableIDAssigner:
    """
    Converts noisy YOLO track IDs into stable project-level IDs.

    Main fix:
      car -> truck -> car class flicker should not create new project IDs.
    """

    def __init__(self, iou_thresh=0.15, dist_thresh=0.90, max_age=8):
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh
        self.max_age = max_age

        self.next_id = 1
        self.raw_to_stable = {}
        self.tracks = {}

    def _new_track(self, frame_idx, raw_id, class_name, bbox):
        stable_id = self.next_id
        self.next_id += 1

        self.tracks[stable_id] = {
            "bbox": [float(x) for x in bbox],
            "last_frame": int(frame_idx),
            "superclass": track_superclass(class_name),
            "class_votes": {str(class_name): 1},
            "raw_ids": {int(raw_id)},
        }

        self.raw_to_stable[int(raw_id)] = stable_id
        return stable_id

    def _display_class(self, stable_id):
        votes = self.tracks[stable_id]["class_votes"]
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def assign(self, frame_idx, raw_id, class_name, bbox):
        raw_id = int(raw_id)
        superclass = track_superclass(class_name)
        bbox = [float(x) for x in bbox]

        # First, trust existing raw-id mapping if it is not too stale.
        if raw_id in self.raw_to_stable:
            sid = self.raw_to_stable[raw_id]
            if sid in self.tracks:
                age = frame_idx - self.tracks[sid]["last_frame"]
                if age <= self.max_age:
                    self._update(sid, frame_idx, raw_id, class_name, bbox)
                    return sid, self._display_class(sid)

        # Otherwise, match against recent tracks using superclass + bbox continuity.
        best_sid = None
        best_score = -1.0

        for sid, tr in self.tracks.items():
            age = frame_idx - tr["last_frame"]
            if age < 0 or age > self.max_age:
                continue

            if tr["superclass"] != superclass:
                continue

            iou = bbox_iou(bbox, tr["bbox"])
            dist_norm = bbox_center_distance_norm(bbox, tr["bbox"])

            # Good match if overlap exists or center motion is reasonable.
            if iou < self.iou_thresh and dist_norm > self.dist_thresh:
                continue

            score = iou + max(0.0, 1.0 - dist_norm) * 0.35 - 0.02 * age

            if score > best_score:
                best_score = score
                best_sid = sid

        if best_sid is None:
            best_sid = self._new_track(frame_idx, raw_id, class_name, bbox)
        else:
            self.raw_to_stable[raw_id] = best_sid
            self._update(best_sid, frame_idx, raw_id, class_name, bbox)

        return best_sid, self._display_class(best_sid)

    def _update(self, stable_id, frame_idx, raw_id, class_name, bbox):
        tr = self.tracks[stable_id]

        tr["bbox"] = [float(x) for x in bbox]
        tr["last_frame"] = int(frame_idx)
        tr["raw_ids"].add(int(raw_id))

        class_name = str(class_name)
        tr["class_votes"][class_name] = tr["class_votes"].get(class_name, 0) + 1

        # If a vehicle flips between car/truck/bus, keep superclass as vehicle.
        tr["superclass"] = track_superclass(class_name)



def safe_model_name(model_name: str):
    return model_name.replace("/", "_").replace("-", "_")


def count_lines(path: Path):
    if not path.exists():
        return 0
    with path.open("r") as f:
        return sum(1 for _ in f)


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


def ensure_yolo_tracks(args, image_paths, n):
    track_path = Path(args.tracks) if args.tracks else TRACK_ROOT / f"tracks_{args.date}_drive_{args.drive}.jsonl"

    needs_tracks = (
        args.overwrite_tracks
        or not track_path.exists()
        or count_lines(track_path) < n
    )

    if needs_tracks:
        print("YOLO track cache missing/incomplete. Creating it now:", track_path)

        cmd = [
            sys.executable,
            "scripts/16_yolo_tracking_video.py",
            "--date", args.date,
            "--drive", args.drive,
            "--model", args.yolo_model,
            "--tracker", args.tracker,
            "--conf", str(args.yolo_conf),
            "--iou", str(args.yolo_iou),
            "--classes", "0", "2", "5", "7",
        ]

        if args.max_frames is not None:
            cmd += ["--max-frames", str(args.max_frames)]

        subprocess.run(cmd, check=True)
    else:
        print("YOLO track cache found:", track_path)

    return track_path


def ensure_da2_depth_cache(args, n):
    depth_cache = (
        Path(args.depth_cache)
        if args.depth_cache
        else DEPTH_CACHE_ROOT / args.date / f"drive_{args.drive}" / "da2_metric_vkitti_vits"
    )

    depth_cache.mkdir(parents=True, exist_ok=True)

    missing = []
    for i in range(n):
        p = depth_cache / f"{i:010d}.npy"
        if args.overwrite_depth_cache or not p.exists():
            missing.append(i)

    if missing:
        print(f"DA2 depth cache missing/incomplete: {len(missing)} frames. Creating it now.")

        cmd = [
            sys.executable,
            "scripts/15_da2_occupancy_grid_video.py",
            "--date", args.date,
            "--drive", args.drive,
            "--mono-y-max", str(args.mono_y_max),
            "--density-scale", "3.0",
        ]

        if args.max_frames is not None:
            cmd += ["--max-frames", str(args.max_frames)]

        if args.overwrite_depth_cache:
            cmd += ["--overwrite-cache"]

        subprocess.run(cmd, check=True)
    else:
        print("DA2 depth cache found:", depth_cache)

    return depth_cache


def ensure_seg_cache(args, n):
    model_key = safe_model_name(args.seg_model)
    seg_cache = SEG_CACHE_ROOT / args.date / f"drive_{args.drive}" / model_key
    seg_cache.mkdir(parents=True, exist_ok=True)

    missing = []
    for i in range(n):
        p = seg_cache / f"{i:010d}.npy"
        if args.overwrite_seg_cache or not p.exists():
            missing.append(i)

    if missing:
        print(f"Road segmentation cache missing/incomplete: {len(missing)} frames. Creating it now.")

        cmd = [
            sys.executable,
            "scripts/18_road_segmentation_cache_video.py",
            "--date", args.date,
            "--drive", args.drive,
            "--model", args.seg_model,
            "--dilate-road-px", str(args.dilate_road_px),
        ]

        if args.max_frames is not None:
            cmd += ["--max-frames", str(args.max_frames)]

        if args.overwrite_seg_cache:
            cmd += ["--overwrite-cache"]

        subprocess.run(cmd, check=True)
    else:
        print("Road segmentation cache found:", seg_cache)

    return seg_cache


def load_tracks_jsonl(path: Path):
    frames = {}
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            frames[int(rec["frame_idx"])] = rec.get("objects", [])
    return frames


def dilate_binary(mask, radius_px):
    mask = mask.astype(np.uint8)
    if radius_px <= 0:
        return mask
    k = 2 * radius_px + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def color_for_track(track_id):
    if track_id is None:
        return (80, 80, 255)
    rng = np.random.default_rng(int(track_id) + 12345)
    color = rng.integers(80, 255, size=3).tolist()
    return tuple(int(c) for c in color)


def get_bottom_center_crop(mask, bbox, radius_px):
    h, w = mask.shape
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    u = int(0.5 * (x1 + x2))
    v = int(y2 - 0.05 * bh)

    r = max(radius_px, int(0.05 * max(bw, bh)))

    x0 = max(0, u - r)
    x3 = min(w, u + r + 1)
    y0 = max(0, v - r)
    y3 = min(h, v + r + 1)

    if x3 <= x0 or y3 <= y0:
        return None

    return mask[y0:y3, x0:x3]


def bbox_touches_mask(mask, bbox, radius_px, min_ratio=0.02):
    crop = get_bottom_center_crop(mask, bbox, radius_px)
    if crop is None or crop.size == 0:
        return False
    return float(np.mean(crop > 0)) >= min_ratio


def keep_detection_by_road_gate(det, label_mask, args):
    class_id = int(det.get("class_id", -1))
    class_name = str(det.get("class_name", "")).lower()

    if class_id not in ALLOWED_CLASS_IDS:
        return False, "class_filtered"

    bbox = det["bbox_xyxy"]

    road = (label_mask == 1).astype(np.uint8)
    sidewalk = (label_mask == 2).astype(np.uint8)

    road_d = dilate_binary(road, args.dilate_road_px)
    sidewalk_d = dilate_binary(sidewalk, args.dilate_sidewalk_px)

    if class_name in VEHICLE_CLASSES:
        if args.disable_road_gate:
            return True, "vehicle_gate_disabled"

        ok = bbox_touches_mask(
            road_d,
            bbox,
            radius_px=args.gate_radius_px,
            min_ratio=args.gate_min_ratio,
        )
        return ok, "vehicle_on_road" if ok else "vehicle_not_on_road"

    if class_name in PERSON_CLASSES:
        if args.person_gate == "none":
            return True, "person_keep_all"

        road_or_sidewalk = np.maximum(road_d, sidewalk_d)
        ok = bbox_touches_mask(
            road_or_sidewalk,
            bbox,
            radius_px=args.gate_radius_px,
            min_ratio=args.gate_min_ratio,
        )
        return ok, "person_near_road_sidewalk" if ok else "person_not_near_road_sidewalk"

    return False, "class_filtered"


def sample_object_depth(depth, bbox_xyxy, min_depth=1.0, max_depth=80.0):
    h, w = depth.shape
    x1, y1, x2, y2 = bbox_xyxy

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    sx1 = int(x1 + 0.35 * bw)
    sx2 = int(x1 + 0.65 * bw)
    sy1 = int(y1 + 0.55 * bh)
    sy2 = int(y1 + 0.90 * bh)

    sx1 = np.clip(sx1, 0, w - 1)
    sx2 = np.clip(sx2, 0, w - 1)
    sy1 = np.clip(sy1, 0, h - 1)
    sy2 = np.clip(sy2, 0, h - 1)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    patch = depth[sy1:sy2, sx1:sx2]
    vals = patch[np.isfinite(patch)]
    vals = vals[(vals >= min_depth) & (vals <= max_depth)]

    if vals.size < 10:
        return None

    return float(np.median(vals))


def project_object_to_camera(bbox_xyxy, depth_m, fx, fy, cx, cy):
    x1, y1, x2, y2 = bbox_xyxy

    u = 0.5 * (x1 + x2)
    v = y1 + 0.80 * (y2 - y1)

    Z = depth_m
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return float(X), float(Y), float(Z)


def grid_shape():
    h = int((FORWARD_MAX - FORWARD_MIN) / BEV_RES)
    w = int((SIDE_MAX - SIDE_MIN) / BEV_RES)
    return h, w


def bev_coords_from_camera(X, Z):
    if Z < FORWARD_MIN or Z > FORWARD_MAX or X < SIDE_MIN or X > SIDE_MAX:
        return None

    h, w = grid_shape()

    row = int((FORWARD_MAX - Z) / BEV_RES)
    col = int((X - SIDE_MIN) / BEV_RES)

    row = int(np.clip(row, 0, h - 1))
    col = int(np.clip(col, 0, w - 1))

    return row, col


def road_mask_to_bev(depth, label_mask, fx, fy, cx, cy, stride):
    h, w = depth.shape
    gh, gw = grid_shape()

    road_mask = label_mask == 1

    y_start = int(0.30 * h)

    ys, xs = np.meshgrid(
        np.arange(y_start, h, stride),
        np.arange(0, w, stride),
        indexing="ij",
    )

    keep = road_mask[ys, xs]
    if not np.any(keep):
        return np.zeros((gh, gw), dtype=np.float32)

    z = depth[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z / fx

    z = z[keep]
    x = x[keep]

    valid = (
        np.isfinite(z)
        & (z >= FORWARD_MIN)
        & (z <= FORWARD_MAX)
        & (x >= SIDE_MIN)
        & (x <= SIDE_MAX)
    )

    z = z[valid]
    x = x[valid]

    road_bev = np.zeros((gh, gw), dtype=np.float32)

    if z.size == 0:
        return road_bev

    row = ((FORWARD_MAX - z) / BEV_RES).astype(np.int32)
    col = ((x - SIDE_MIN) / BEV_RES).astype(np.int32)

    row = np.clip(row, 0, gh - 1)
    col = np.clip(col, 0, gw - 1)

    np.add.at(road_bev, (row, col), 1.0)

    if road_bev.max() > 0:
        road_bev = np.log1p(road_bev)
        road_bev = road_bev / road_bev.max()

    road_bev = cv2.GaussianBlur(road_bev, (5, 5), 0)
    road_bev = np.clip(road_bev, 0.0, 1.0)

    return road_bev.astype(np.float32)


def object_risk_score(X, Z, class_name, prev_state=None):
    distance_risk = 1.0 - np.clip((Z - FORWARD_MIN) / (FORWARD_MAX - FORWARD_MIN), 0.0, 1.0)
    center_risk = 1.0 - np.clip(abs(X) / 8.0, 0.0, 1.0)

    approach_rate = 0.0
    lateral_centering_rate = 0.0
    motion_weight = 1.0

    if prev_state is not None:
        prev_x = prev_state["X"]
        prev_z = prev_state["Z"]

        approach_rate = prev_z - Z
        lateral_centering_rate = abs(prev_x) - abs(X)

        if approach_rate > 0.15:
            motion_weight += 0.15
        if lateral_centering_rate > 0.08:
            motion_weight += 0.10

    class_weight = CLASS_RISK_WEIGHTS.get(class_name, 1.0)

    risk = 0.55 * distance_risk + 0.35 * center_risk + 0.10 * max(approach_rate, 0.0)
    risk = risk * class_weight * motion_weight

    return float(np.clip(risk, 0.0, 1.0)), float(approach_rate), float(lateral_centering_rate)


def smooth_track_position(track_state, track_id, X, Y, Z, depth_m, alpha):
    prev = track_state.get(track_id)

    if prev is None or alpha <= 0.0:
        smoothed = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "depth": depth_m,
        }
    else:
        smoothed = {
            "X": alpha * prev["X"] + (1.0 - alpha) * X,
            "Y": alpha * prev["Y"] + (1.0 - alpha) * Y,
            "Z": alpha * prev["Z"] + (1.0 - alpha) * Z,
            "depth": alpha * prev["depth"] + (1.0 - alpha) * depth_m,
        }

    track_state[track_id] = smoothed
    return prev, smoothed


def draw_rect_object_occupancy(occ, obj, footprint_scale):
    coords = bev_coords_from_camera(obj["bev_x_m"], obj["bev_z_m"])
    if coords is None:
        return

    row, col = coords
    class_name = obj["class_name"]
    width_m, length_m = CLASS_FOOTPRINTS.get(class_name, (2.0, 3.0))

    width_m *= footprint_scale
    length_m *= footprint_scale

    half_w = max(1, int((width_m * 0.5) / BEV_RES))
    half_l = max(1, int((length_m * 0.5) / BEV_RES))

    gh, gw = occ.shape

    r1 = max(0, row - half_l)
    r2 = min(gh - 1, row + half_l)
    c1 = max(0, col - half_w)
    c2 = min(gw - 1, col + half_w)

    value = float(np.clip(0.45 + 0.55 * obj["risk"], 0.0, 1.0))
    cv2.rectangle(occ, (c1, r1), (c2, r2), value, thickness=-1)


def object_occ_to_risk(occ):
    h, w = occ.shape

    closeness = np.linspace(0.15, 1.0, h).reshape(h, 1)

    cols = np.arange(w)
    center = w / 2.0
    dist = np.abs(cols - center) / center

    center_weight = 1.0 - 0.75 * dist
    center_weight = np.clip(center_weight, 0.25, 1.0).reshape(1, w)

    risk = occ * closeness * center_weight
    risk = cv2.GaussianBlur(risk, (7, 7), 0)

    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def colorize_map(x, cmap):
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cmap)


def make_road_object_bev_color(road_bev, obj_occ):
    road = np.clip(road_bev, 0.0, 1.0)
    obj = np.clip(obj_occ, 0.0, 1.0)

    h, w = road.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Road layer: green-ish background.
    out[:, :, 1] = np.maximum(out[:, :, 1], (road * 130).astype(np.uint8))
    out[:, :, 0] = np.maximum(out[:, :, 0], (road * 40).astype(np.uint8))

    # Object layer: red/yellow stronger occupancy.
    out[:, :, 2] = np.maximum(out[:, :, 2], (obj * 255).astype(np.uint8))
    out[:, :, 1] = np.maximum(out[:, :, 1], (obj * 160).astype(np.uint8))

    return out


def put_map_in_panel(map_color, panel_w, panel_h):
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


def overlay_road_sidewalk(rgb_bgr, label_mask, alpha=0.40):
    out = rgb_bgr.copy()

    color = np.zeros_like(rgb_bgr)
    color[label_mask == 1] = (60, 210, 60)      # road
    color[label_mask == 2] = (220, 180, 70)     # sidewalk

    mask = label_mask > 0
    out[mask] = (
        (1.0 - alpha) * out[mask].astype(np.float32)
        + alpha * color[mask].astype(np.float32)
    ).astype(np.uint8)

    return out


def draw_rgb_objects(rgb, objects):
    out = rgb.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        tid = obj["track_id"]
        cls = obj["class_name"]
        dist = obj["depth_m"]
        risk = obj["risk"]

        color = color_for_track(tid)

        cv2.rectangle(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )

        label = f"ID {tid} {cls} {dist:.1f}m R:{risk:.2f}"

        cv2.putText(
            out,
            label,
            (int(x1), max(25, int(y1) - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_bev_labels(bev_color, objects):
    out = bev_color.copy()

    for obj in objects:
        coords = bev_coords_from_camera(obj["bev_x_m"], obj["bev_z_m"])
        if coords is None:
            continue

        row, col = coords
        tid = obj["track_id"]
        cls = obj["class_name"]
        dist = obj["depth_m"]

        cv2.circle(out, (col, row), 3, (255, 255, 255), -1)

        label = f"{tid}:{cls} {dist:.1f}m"

        cv2.putText(
            out,
            label,
            (col + 4, row - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def make_panel(rgb_bgr, label_mask, objects, road_bev, obj_occ, risk_grid, frame_idx, date, drive):
    top_tile_w = 621
    top_tile_h = 188

    bottom_panel_w = 621
    bottom_panel_h = 376

    overlay = overlay_road_sidewalk(rgb_bgr, label_mask)
    rgb_objects = draw_rgb_objects(rgb_bgr, objects)

    overlay_tile = cv2.resize(overlay, (top_tile_w, top_tile_h))
    rgb_obj_tile = cv2.resize(rgb_objects, (top_tile_w, top_tile_h))

    bev_color = make_road_object_bev_color(road_bev, obj_occ)
    bev_color = draw_bev_labels(bev_color, objects)

    risk_color = colorize_map(risk_grid, cv2.COLORMAP_JET)
    risk_color = draw_bev_labels(risk_color, objects)

    bev_panel = put_map_in_panel(bev_color, bottom_panel_w, bottom_panel_h)
    risk_panel = put_map_in_panel(risk_color, bottom_panel_w, bottom_panel_h)

    top = np.concatenate([overlay_tile, rgb_obj_tile], axis=1)
    bottom = np.concatenate([bev_panel, risk_panel], axis=1)
    panel = np.concatenate([top, bottom], axis=0)

    cv2.putText(panel, f"Road/sidewalk segmentation | frame {frame_idx:04d}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Road-gated tracked objects",
                (top_tile_w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Projected road + rectangular object occupancy",
                (15, top_tile_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Temporal object-aware risk",
                (top_tile_w + 15, top_tile_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, f"KITTI {date} drive {drive} | green=road, red/yellow=objects",
                (15, top_tile_h + bottom_panel_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,180,180), 1, cv2.LINE_AA)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--tracks", type=str, default=None)
    parser.add_argument("--depth-cache", type=str, default=None)

    parser.add_argument("--yolo-model", type=str, default="yolov8m.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.5)

    parser.add_argument("--seg-model", type=str, default="nvidia/segformer-b0-finetuned-cityscapes-768-768")

    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=60.0)

    parser.add_argument("--mono-y-max", type=float, default=1.35)

    parser.add_argument("--dilate-road-px", type=int, default=12)
    parser.add_argument("--dilate-sidewalk-px", type=int, default=10)
    parser.add_argument("--gate-radius-px", type=int, default=10)
    parser.add_argument("--gate-min-ratio", type=float, default=0.02)
    parser.add_argument("--disable-road-gate", action="store_true")
    parser.add_argument("--person-gate", type=str, default="road_sidewalk", choices=["road_sidewalk", "none"])

    parser.add_argument("--road-bev-stride", type=int, default=5)
    parser.add_argument("--footprint-scale", type=float, default=0.75)

    parser.add_argument("--obj-smooth-alpha", type=float, default=0.65)
    parser.add_argument("--grid-smooth-alpha", type=float, default=0.35)

    parser.add_argument("--disable-stable-ids", action="store_true")
    parser.add_argument("--stable-iou-thresh", type=float, default=0.15)
    parser.add_argument("--stable-dist-thresh", type=float, default=0.90)
    parser.add_argument("--stable-max-age", type=int, default=8)

    parser.add_argument("--overwrite-tracks", action="store_true")
    parser.add_argument("--overwrite-depth-cache", action="store_true")
    parser.add_argument("--overwrite-seg-cache", action="store_true")

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

    print("Sequence:", seq_root)
    print("Frames used:", n)
    print("Only classes: person, car, bus, truck")
    print("Vehicle road gate:", not args.disable_road_gate)
    print("Person gate:", args.person_gate)
    print("Object smoothing alpha:", args.obj_smooth_alpha)
    print("Grid smoothing alpha:", args.grid_smooth_alpha)
    print("Stable IDs:", not args.disable_stable_ids)

    track_path = ensure_yolo_tracks(args, image_paths, n)
    depth_cache = ensure_da2_depth_cache(args, n)
    seg_cache = ensure_seg_cache(args, n)

    tracks_by_frame = load_tracks_jsonl(track_path)
    fx, fy, cx, cy = read_kitti_intrinsics(calib_path)

    out_video = OUTPUT_VIDEO_DIR / f"phase10b_road_gated_object_bev_risk_{args.date}_drive_{args.drive}.mp4"
    out_jsonl = OUTPUT_OBJECT_DIR / f"road_gated_objects_da2_{args.date}_drive_{args.drive}.jsonl"

    panel_w = 621 * 2
    panel_h = 188 + 376

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_w, panel_h),
    )

    track_state = {}
    prev_obj_occ = None

    stable_assigner = None
    if not args.disable_stable_ids:
        stable_assigner = StableIDAssigner(
            iou_thresh=args.stable_iou_thresh,
            dist_thresh=args.stable_dist_thresh,
            max_age=args.stable_max_age,
        )

    with out_jsonl.open("w") as f:
        for frame_idx in tqdm(range(n)):
            rgb = cv2.imread(str(image_paths[frame_idx]))
            if rgb is None:
                continue

            depth_path = depth_cache / f"{frame_idx:010d}.npy"
            seg_path = seg_cache / f"{frame_idx:010d}.npy"

            if not depth_path.exists():
                raise FileNotFoundError(depth_path)
            if not seg_path.exists():
                raise FileNotFoundError(seg_path)

            depth = np.load(depth_path).astype(np.float32)
            label_mask = np.load(seg_path).astype(np.uint8)

            img_h, img_w = rgb.shape[:2]

            if depth.shape[:2] != (img_h, img_w):
                depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            if label_mask.shape[:2] != (img_h, img_w):
                label_mask = cv2.resize(label_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            frame_tracks = tracks_by_frame.get(frame_idx, [])
            objects = []
            reject_counts = {}

            for det in frame_tracks:
                if float(det.get("confidence", 0.0)) < args.min_conf:
                    continue

                track_id = det.get("track_id")
                if track_id is None:
                    continue

                class_id = int(det.get("class_id", -1))
                class_name = str(det.get("class_name", "")).lower()

                if class_id not in ALLOWED_CLASS_IDS:
                    continue

                keep, reason = keep_detection_by_road_gate(det, label_mask, args)
                if not keep:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                    continue

                bbox = det["bbox_xyxy"]

                depth_m = sample_object_depth(
                    depth,
                    bbox,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                )

                if depth_m is None:
                    reject_counts["bad_depth"] = reject_counts.get("bad_depth", 0) + 1
                    continue

                X, Y, Z = project_object_to_camera(
                    bbox,
                    depth_m,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )

                if Z < FORWARD_MIN or Z > FORWARD_MAX or X < SIDE_MIN or X > SIDE_MAX:
                    reject_counts["outside_bev"] = reject_counts.get("outside_bev", 0) + 1
                    continue

                raw_yolo_track_id = int(track_id)

                if stable_assigner is not None:
                    tid, stable_class_name = stable_assigner.assign(
                        frame_idx=frame_idx,
                        raw_id=raw_yolo_track_id,
                        class_name=class_name,
                        bbox=bbox,
                    )
                    class_name_for_output = stable_class_name
                else:
                    tid = raw_yolo_track_id
                    class_name_for_output = class_name

                prev_state, smooth_state = smooth_track_position(
                    track_state=track_state,
                    track_id=tid,
                    X=X,
                    Y=Y,
                    Z=Z,
                    depth_m=depth_m,
                    alpha=args.obj_smooth_alpha,
                )

                Xs = smooth_state["X"]
                Ys = smooth_state["Y"]
                Zs = smooth_state["Z"]
                Ds = smooth_state["depth"]

                risk, approach_rate, lateral_rate = object_risk_score(
                    Xs,
                    Zs,
                    class_name_for_output,
                    prev_state=prev_state,
                )

                obj = {
                    "frame_idx": frame_idx,
                    "track_id": tid,
                    "raw_yolo_track_id": raw_yolo_track_id,
                    "class_id": class_id,
                    "raw_class_name": class_name,
                    "class_name": class_name_for_output,
                    "confidence": float(det["confidence"]),
                    "bbox_xyxy": [float(x) for x in bbox],
                    "gate_reason": reason,

                    "raw_depth_m": float(depth_m),
                    "raw_bev_x_m": float(X),
                    "raw_bev_z_m": float(Z),

                    "depth_m": float(Ds),
                    "bev_x_m": float(Xs),
                    "bev_y_cam_m": float(Ys),
                    "bev_z_m": float(Zs),

                    "risk": float(risk),
                    "approach_rate_m_per_frame": float(approach_rate),
                    "lateral_centering_rate_m_per_frame": float(lateral_rate),
                }

                objects.append(obj)

            road_bev = road_mask_to_bev(
                depth=depth,
                label_mask=label_mask,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                stride=args.road_bev_stride,
            )

            gh, gw = grid_shape()
            obj_occ = np.zeros((gh, gw), dtype=np.float32)

            for obj in objects:
                draw_rect_object_occupancy(obj_occ, obj, footprint_scale=args.footprint_scale)

            obj_occ = cv2.GaussianBlur(obj_occ, (3, 3), 0)
            obj_occ = np.clip(obj_occ, 0.0, 1.0)

            if prev_obj_occ is not None and args.grid_smooth_alpha > 0.0:
                obj_occ = (
                    args.grid_smooth_alpha * prev_obj_occ
                    + (1.0 - args.grid_smooth_alpha) * obj_occ
                )
                obj_occ = np.clip(obj_occ, 0.0, 1.0)

            prev_obj_occ = obj_occ.copy()

            risk_grid = object_occ_to_risk(obj_occ)

            panel = make_panel(
                rgb_bgr=rgb,
                label_mask=label_mask,
                objects=objects,
                road_bev=road_bev,
                obj_occ=obj_occ,
                risk_grid=risk_grid,
                frame_idx=frame_idx,
                date=args.date,
                drive=args.drive,
            )

            writer.write(panel)

            f.write(json.dumps({
                "frame_idx": frame_idx,
                "objects": objects,
                "reject_counts": reject_counts,
            }) + "\n")

    writer.release()

    print("Saved video:", out_video)
    print("Saved object records:", out_jsonl)


if __name__ == "__main__":
    main()
