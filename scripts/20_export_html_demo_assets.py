#!/usr/bin/env python3
"""
Export a static HTML-only interactive demo for liftdepth-risk.

Reads precomputed project caches:
  - KITTI RGB frames
  - DA2 depth cache
  - road segmentation cache
  - road-gated object records from script 19

Exports:
  demo/<sequence>/
    index.html
    metadata.json
    data/rgb/*.jpg
    data/depth/*.jpg
    data/road_overlay/*.jpg
    data/road_bev/*.png
    data/object_occ/*.png
    data/risk/*.png
    data/frame_metadata/*.json

Open:
  xdg-open demo/kitti_2011_09_26_drive_0020/index.html
"""

from pathlib import Path
import argparse
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm


DATA_ROOT = Path("data/kitti/raw")
DEPTH_CACHE_ROOT = Path("outputs/depth_cache/da2")
SEG_CACHE_ROOT = Path("outputs/seg_cache/road_seg")
OBJECT_ROOT = Path("outputs/objects")
DEMO_ROOT = Path("demo")

FORWARD_MIN = 1.0
FORWARD_MAX = 50.0
SIDE_MIN = -18.0
SIDE_MAX = 18.0
BEV_RES = 0.20

CLASS_FOOTPRINTS = {
    "person": (0.8, 0.8),
    "car": (2.0, 4.2),
    "bus": (2.8, 9.0),
    "truck": (2.8, 7.0),
}


def safe_model_name(model_name: str):
    return model_name.replace("/", "_").replace("-", "_")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def load_object_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Object metadata not found: {path}\n"
            "Run scripts/19_road_gated_object_bev_risk.py first."
        )

    frames = {}
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            frames[int(rec["frame_idx"])] = rec

    return frames


def colorize_depth(depth_m: np.ndarray, max_depth: float = 80.0):
    depth = depth_m.astype(np.float32)
    depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth = np.clip(depth, 0.0, max_depth)

    inv = 1.0 - depth / max_depth
    inv = np.clip(inv, 0.0, 1.0)

    u8 = (inv * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)


def overlay_road_sidewalk(rgb_bgr, label_mask, alpha=0.42):
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
    road_bev = np.zeros((gh, gw), dtype=np.float32)

    if not np.any(keep):
        return road_bev

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
    return np.clip(road_bev, 0.0, 1.0).astype(np.float32)


def draw_rect_object_occupancy(occ, obj, footprint_scale=0.65):
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

    value = float(np.clip(0.45 + 0.55 * obj.get("risk", 0.5), 0.0, 1.0))
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


def make_road_bev_color(road_bev):
    road = np.clip(road_bev, 0.0, 1.0)
    h, w = road.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 1] = (road * 170).astype(np.uint8)
    out[:, :, 0] = (road * 55).astype(np.uint8)
    return out


def make_object_occ_color(obj_occ):
    obj = np.clip(obj_occ, 0.0, 1.0)
    h, w = obj.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 2] = (obj * 255).astype(np.uint8)
    out[:, :, 1] = (obj * 180).astype(np.uint8)
    return out


def write_img(path: Path, img_bgr, jpg_quality=88):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    else:
        cv2.imwrite(str(path), img_bgr)


def sanitize_objects_for_html(objects):
    clean = []
    for obj in objects:
        clean.append({
            "frame_idx": int(obj.get("frame_idx", 0)),
            "track_id": int(obj.get("track_id", -1)),
            "class_name": str(obj.get("class_name", "object")),
            "confidence": float(obj.get("confidence", 0.0)),
            "bbox_xyxy": [float(x) for x in obj.get("bbox_xyxy", [0, 0, 0, 0])],
            "depth_m": float(obj.get("depth_m", 0.0)),
            "bev_x_m": float(obj.get("bev_x_m", 0.0)),
            "bev_z_m": float(obj.get("bev_z_m", 0.0)),
            "risk": float(obj.get("risk", 0.0)),
            "gate_reason": str(obj.get("gate_reason", "")),
            "approach_rate_m_per_frame": float(obj.get("approach_rate_m_per_frame", 0.0)),
            "lateral_centering_rate_m_per_frame": float(obj.get("lateral_centering_rate_m_per_frame", 0.0)),
        })
    return clean


def build_index_html(inline_metadata_json: str, inline_frames_json: str):
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>LiftDepth Risk Demo</title>
  <style>
    :root {
      --bg: #071018;
      --panel: #0d1824;
      --panel2: #111f2d;
      --text: #e8f0f7;
      --muted: #8fa4b8;
      --line: rgba(255,255,255,0.10);
      --green: #5ee787;
      --yellow: #ffd166;
      --red: #ff5d73;
      --blue: #7cc7ff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(72, 128, 255, 0.16), transparent 36rem),
        radial-gradient(circle at top right, rgba(94, 231, 135, 0.10), transparent 34rem),
        var(--bg);
      color: var(--text);
    }

    header {
      padding: 22px 26px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(7,16,24,0.72);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 20;
    }

    h1 {
      font-size: 24px;
      margin: 0 0 6px;
      letter-spacing: -0.02em;
    }

    .subtitle {
      color: var(--muted);
      font-size: 14px;
      max-width: 980px;
      line-height: 1.45;
    }

    .badge-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }

    .badge {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.05);
      color: #dbe8f4;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 12px;
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 18px;
      padding: 18px;
      max-width: 1780px;
      margin: 0 auto;
    }

    .controls {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(13,24,36,0.80);
      margin-bottom: 16px;
    }

    .control-top {
      display: grid;
      grid-template-columns: auto minmax(220px, 1fr) auto auto auto auto;
      gap: 12px;
      align-items: center;
    }

    button, select {
      background: var(--panel2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px 12px;
      font-size: 14px;
      cursor: pointer;
    }

    button:hover, select:hover {
      border-color: rgba(255,255,255,0.22);
      background: #15283a;
    }

    input[type="range"] {
      width: 100%;
    }

    .frame-readout {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .toggles {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }

    label { user-select: none; }

    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: rgba(13,24,36,0.78);
      box-shadow: 0 10px 30px rgba(0,0,0,0.20);
      min-width: 0;
    }

    .card h2 {
      margin: 0;
      padding: 11px 13px;
      font-size: 14px;
      color: #dfeaf4;
      background: rgba(255,255,255,0.035);
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }

    .card h2 span {
      color: var(--muted);
      font-weight: 500;
      font-size: 12px;
    }

    .img-wrap {
      position: relative;
      background: #000;
      aspect-ratio: 1242 / 375;
    }

    .bev-wrap {
      aspect-ratio: 180 / 245;
      max-height: 420px;
    }

    .img-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    canvas.overlay {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }

    aside {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(13,24,36,0.82);
      height: calc(100vh - 128px);
      position: sticky;
      top: 104px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .side-section {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }

    .side-section h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 9px;
      background: rgba(255,255,255,0.035);
    }

    .metric .k {
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }

    .metric .v {
      font-size: 17px;
      font-weight: 700;
    }

    .object-list {
      padding: 10px 14px 14px;
      overflow: auto;
      flex: 1;
    }

    .obj {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      margin-bottom: 10px;
      background: rgba(255,255,255,0.035);
    }

    .obj-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }

    .obj-name { font-weight: 750; }

    .risk {
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(255,255,255,0.08);
    }

    .risk.low { color: var(--green); }
    .risk.med { color: var(--yellow); }
    .risk.high { color: var(--red); }

    .obj-detail {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }

    .rejects {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
    }

    .story {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(13,24,36,0.72);
      padding: 16px 18px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }

    .story strong { color: var(--text); }

    @media (max-width: 1180px) {
      main { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
      .control-top { grid-template-columns: 1fr; }
      header { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <h1 id="title">LiftDepth Risk Demo</h1>
    <div class="subtitle">
      A camera-only perception pipeline: RGB → DA2 metric depth → road segmentation → tracked road users → BEV occupancy → object-aware risk.
    </div>
    <div class="badge-row">
      <div class="badge">DA2 metric depth</div>
      <div class="badge">Road segmentation</div>
      <div class="badge">YOLO tracking</div>
      <div class="badge">BEV occupancy</div>
      <div class="badge">Risk heatmap</div>
    </div>
  </header>

  <main>
    <section class="viewer">
      <div class="controls">
        <div class="control-top">
          <button id="playBtn">Play</button>
          <input id="frameSlider" type="range" min="0" max="0" value="0" />
          <div class="frame-readout" id="frameReadout">Frame 0 / 0</div>
          <select id="speedSelect">
            <option value="500">0.5×</option>
            <option value="250" selected>1×</option>
            <option value="125">2×</option>
            <option value="80">Fast</option>
          </select>
          <button id="prevBtn">Prev</button>
          <button id="nextBtn">Next</button>
        </div>

        <div class="toggles">
          <label><input type="checkbox" id="showBoxes" checked /> show RGB boxes</label>
          <label><input type="checkbox" id="showLabels" checked /> show labels</label>
          <label><input type="checkbox" id="showBevLabels" checked /> show BEV IDs</label>
          <label><input type="checkbox" id="showRejected" /> show rejected counts</label>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <h2>RGB Input + Live Tracks <span>drawn in browser</span></h2>
          <div class="img-wrap">
            <img id="rgbImg" alt="RGB frame" />
            <canvas id="rgbCanvas" class="overlay"></canvas>
          </div>
        </div>

        <div class="card">
          <h2>DA2 Metric Depth <span>geometry source</span></h2>
          <div class="img-wrap">
            <img id="depthImg" alt="Depth map" />
          </div>
        </div>

        <div class="card">
          <h2>Road / Sidewalk Segmentation <span>context gate</span></h2>
          <div class="img-wrap">
            <img id="roadOverlayImg" alt="Road overlay" />
          </div>
        </div>

        <div class="card">
          <h2>Projected Road BEV <span>green = road + projected objects</span></h2>
          <div class="img-wrap bev-wrap">
            <img id="roadBevImg" alt="Projected road BEV" />
            <canvas id="roadBevCanvas" class="overlay"></canvas>
          </div>
        </div>

        <div class="card">
          <h2>Object Occupancy Grid <span>rectangular road users</span></h2>
          <div class="img-wrap bev-wrap">
            <img id="objOccImg" alt="Object occupancy grid" />
            <canvas id="occCanvas" class="overlay"></canvas>
          </div>
        </div>

        <div class="card">
          <h2>Object-Aware Risk Heatmap <span>distance × center × motion</span></h2>
          <div class="img-wrap bev-wrap">
            <img id="riskImg" alt="Risk heatmap" />
            <canvas id="riskCanvas" class="overlay"></canvas>
          </div>
        </div>
      </div>

      <div class="story">
        <strong>Purpose:</strong> this demo shows how a monocular driving frame becomes a structured BEV scene representation.
        Depth supplies geometry, road segmentation filters context, YOLO tracking gives object identity, and the final occupancy/risk maps focus attention on road-relevant objects.
      </div>
    </section>

    <aside>
      <div class="side-section">
        <h3>Current frame</h3>
        <div class="metric-grid">
          <div class="metric"><div class="k">Frame</div><div class="v" id="metricFrame">0</div></div>
          <div class="metric"><div class="k">Objects</div><div class="v" id="metricObjects">0</div></div>
          <div class="metric"><div class="k">Max risk</div><div class="v" id="metricRisk">0.00</div></div>
          <div class="metric"><div class="k">Sequence</div><div class="v" id="metricSeq">-</div></div>
        </div>
      </div>

      <div class="side-section">
        <h3>Rejected detections</h3>
        <div class="rejects" id="rejects">Toggle “show rejected counts”.</div>
      </div>

      <div class="object-list">
        <h3>Tracked objects</h3>
        <div id="objectList"></div>
      </div>
    </aside>
  </main>

  <script>
    const INLINE_METADATA = __INLINE_METADATA_JSON__;
    const INLINE_FRAME_DATA = __INLINE_FRAMES_JSON__;

    let meta = INLINE_METADATA;
    let frameData = INLINE_FRAME_DATA;
    let current = 0;
    let playing = false;
    let timer = null;

    const els = {
      title: document.getElementById("title"),
      playBtn: document.getElementById("playBtn"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      slider: document.getElementById("frameSlider"),
      speed: document.getElementById("speedSelect"),
      frameReadout: document.getElementById("frameReadout"),
      showBoxes: document.getElementById("showBoxes"),
      showLabels: document.getElementById("showLabels"),
      showBevLabels: document.getElementById("showBevLabels"),
      showRejected: document.getElementById("showRejected"),

      rgbImg: document.getElementById("rgbImg"),
      depthImg: document.getElementById("depthImg"),
      roadOverlayImg: document.getElementById("roadOverlayImg"),
      roadBevImg: document.getElementById("roadBevImg"),
      objOccImg: document.getElementById("objOccImg"),
      riskImg: document.getElementById("riskImg"),

      rgbCanvas: document.getElementById("rgbCanvas"),
      roadBevCanvas: document.getElementById("roadBevCanvas"),
      occCanvas: document.getElementById("occCanvas"),
      riskCanvas: document.getElementById("riskCanvas"),

      metricFrame: document.getElementById("metricFrame"),
      metricObjects: document.getElementById("metricObjects"),
      metricRisk: document.getElementById("metricRisk"),
      metricSeq: document.getElementById("metricSeq"),
      rejects: document.getElementById("rejects"),
      objectList: document.getElementById("objectList"),
    };

    function pad(n) {
      return String(n).padStart(6, "0");
    }

    async function init() {
      els.title.textContent = meta.title || "LiftDepth Risk Demo";
      els.slider.max = meta.num_frames - 1;
      els.metricSeq.textContent = meta.drive || "-";

      await goToFrame(0);

      els.playBtn.addEventListener("click", togglePlay);
      els.prevBtn.addEventListener("click", () => goToFrame(Math.max(0, current - 1)));
      els.nextBtn.addEventListener("click", () => goToFrame(Math.min(meta.num_frames - 1, current + 1)));
      els.slider.addEventListener("input", () => goToFrame(Number(els.slider.value)));

      els.showBoxes.addEventListener("change", redrawOverlays);
      els.showLabels.addEventListener("change", redrawOverlays);
      els.showBevLabels.addEventListener("change", redrawOverlays);
      els.showRejected.addEventListener("change", renderSidebar);

      window.addEventListener("resize", redrawOverlays);

      document.addEventListener("keydown", (e) => {
        if (e.code === "Space") {
          e.preventDefault();
          togglePlay();
        } else if (e.key === "ArrowRight") {
          goToFrame(Math.min(meta.num_frames - 1, current + 1));
        } else if (e.key === "ArrowLeft") {
          goToFrame(Math.max(0, current - 1));
        }
      });
    }

    async function goToFrame(idx) {
      current = idx;
      const id = pad(current);

      els.rgbImg.src = `data/rgb/${id}.jpg`;
      els.depthImg.src = `data/depth/${id}.jpg`;
      els.roadOverlayImg.src = `data/road_overlay/${id}.jpg`;
      els.roadBevImg.src = `data/road_bev/${id}.png`;
      els.objOccImg.src = `data/object_occ/${id}.png`;
      els.riskImg.src = `data/risk/${id}.png`;

      els.slider.value = current;
      els.frameReadout.textContent = `Frame ${current + 1} / ${meta.num_frames}`;

      const drawWhenReady = () => {
        redrawOverlays();
        renderSidebar();
      };

      if (els.rgbImg.complete) drawWhenReady();
      else els.rgbImg.onload = drawWhenReady;
    }

    function togglePlay() {
      playing = !playing;
      els.playBtn.textContent = playing ? "Pause" : "Play";

      if (playing) {
        timer = setInterval(() => {
          const next = current + 1 >= meta.num_frames ? 0 : current + 1;
          goToFrame(next);
        }, Number(els.speed.value));
      } else {
        clearInterval(timer);
        timer = null;
      }
    }

    function resizeCanvasToElement(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w: rect.width, h: rect.height };
    }

    function colorForTrack(id) {
      const x = Math.abs(Math.sin((id + 11) * 12.9898) * 43758.5453);
      const y = Math.abs(Math.sin((id + 23) * 78.233) * 24634.6345);
      const z = Math.abs(Math.sin((id + 37) * 39.425) * 56445.2345);
      return `rgb(${Math.floor(90 + (x % 1) * 165)}, ${Math.floor(90 + (y % 1) * 165)}, ${Math.floor(90 + (z % 1) * 165)})`;
    }

    function fitImageIntoCanvas(canvasW, canvasH, imgW, imgH) {
      const scale = Math.min(canvasW / imgW, canvasH / imgH);
      const drawW = imgW * scale;
      const drawH = imgH * scale;
      const x0 = (canvasW - drawW) / 2;
      const y0 = (canvasH - drawH) / 2;
      return { scale, x0, y0, drawW, drawH };
    }

    function redrawOverlays() {
      drawRgbBoxes();
      drawBevLabels(els.roadBevCanvas);
      drawBevLabels(els.occCanvas);
      drawBevLabels(els.riskCanvas);
    }

    function drawRgbBoxes() {
      const { ctx, w, h } = resizeCanvasToElement(els.rgbCanvas);
      ctx.clearRect(0, 0, w, h);

      if (!els.showBoxes.checked || !meta || !frameData[current]) return;

      const rec = frameData[current];
      const imgW = meta.image_size.width;
      const imgH = meta.image_size.height;
      const fit = fitImageIntoCanvas(w, h, imgW, imgH);

      for (const obj of rec.objects || []) {
        const [x1, y1, x2, y2] = obj.bbox_xyxy;
        const sx1 = fit.x0 + x1 * fit.scale;
        const sy1 = fit.y0 + y1 * fit.scale;
        const sx2 = fit.x0 + x2 * fit.scale;
        const sy2 = fit.y0 + y2 * fit.scale;

        const col = colorForTrack(obj.track_id);
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

        if (els.showLabels.checked) {
          const label = `ID ${obj.track_id} ${obj.class_name} ${obj.depth_m.toFixed(1)}m R:${obj.risk.toFixed(2)}`;
          ctx.font = "12px ui-sans-serif, system-ui";
          const tw = ctx.measureText(label).width;
          const th = 18;
          ctx.fillStyle = "rgba(0,0,0,0.72)";
          ctx.fillRect(sx1, Math.max(0, sy1 - th - 3), tw + 8, th);
          ctx.fillStyle = col;
          ctx.fillText(label, sx1 + 4, Math.max(13, sy1 - 8));
        }
      }
    }

    function drawBevLabels(canvas) {
      const { ctx, w, h } = resizeCanvasToElement(canvas);
      ctx.clearRect(0, 0, w, h);

      if (!els.showBevLabels.checked || !meta || !frameData[current]) return;

      const bevW = meta.bev_size.width * meta.bev_size.render_scale;
      const bevH = meta.bev_size.height * meta.bev_size.render_scale;
      const fit = fitImageIntoCanvas(w, h, bevW, bevH);

      for (const obj of frameData[current].objects || []) {
        const sideMin = meta.bev_bounds.side_min_m;
        const sideMax = meta.bev_bounds.side_max_m;
        const forwardMin = meta.bev_bounds.forward_min_m;
        const forwardMax = meta.bev_bounds.forward_max_m;

        const x = obj.bev_x_m;
        const z = obj.bev_z_m;

        if (z < forwardMin || z > forwardMax || x < sideMin || x > sideMax) continue;

        const col = ((x - sideMin) / (sideMax - sideMin)) * bevW;
        const row = ((forwardMax - z) / (forwardMax - forwardMin)) * bevH;

        const px = fit.x0 + col * fit.scale;
        const py = fit.y0 + row * fit.scale;

        ctx.fillStyle = "white";
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fill();

        if (els.showLabels.checked) {
          const label = `${obj.track_id}:${obj.class_name} ${obj.depth_m.toFixed(1)}m`;
          ctx.font = "11px ui-sans-serif, system-ui";
          ctx.fillStyle = "rgba(0,0,0,0.76)";
          const tw = ctx.measureText(label).width;
          ctx.fillRect(px + 5, py - 16, tw + 7, 16);
          ctx.fillStyle = "white";
          ctx.fillText(label, px + 8, py - 4);
        }
      }
    }

    function riskClass(r) {
      if (r >= 0.65) return "high";
      if (r >= 0.35) return "med";
      return "low";
    }

    function renderSidebar() {
      if (!frameData[current]) return;

      const rec = frameData[current];
      const objs = rec.objects || [];
      const maxRisk = objs.length ? Math.max(...objs.map(o => o.risk || 0)) : 0;

      els.metricFrame.textContent = String(current);
      els.metricObjects.textContent = String(objs.length);
      els.metricRisk.textContent = maxRisk.toFixed(2);

      if (els.showRejected.checked) {
        const rejects = rec.reject_counts || {};
        const lines = Object.keys(rejects).length
          ? Object.entries(rejects).map(([k,v]) => `${k}: ${v}`).join("\n")
          : "No rejected detections recorded.";
        els.rejects.textContent = lines;
      } else {
        els.rejects.textContent = "Toggle “show rejected counts”.";
      }

      if (!objs.length) {
        els.objectList.innerHTML = `<div class="obj-detail">No road-gated tracked objects in this frame.</div>`;
        return;
      }

      const sorted = [...objs].sort((a, b) => (b.risk || 0) - (a.risk || 0));

      els.objectList.innerHTML = sorted.map(obj => {
        const rc = riskClass(obj.risk || 0);
        return `
          <div class="obj">
            <div class="obj-top">
              <div class="obj-name">ID ${obj.track_id} · ${obj.class_name}</div>
              <div class="risk ${rc}">risk ${(obj.risk || 0).toFixed(2)}</div>
            </div>
            <div class="obj-detail">
              distance: ${(obj.depth_m || 0).toFixed(1)} m<br/>
              BEV: x ${(obj.bev_x_m || 0).toFixed(1)} m, z ${(obj.bev_z_m || 0).toFixed(1)} m<br/>
              confidence: ${(obj.confidence || 0).toFixed(2)}<br/>
              gate: ${obj.gate_reason || "-"}
            </div>
          </div>
        `;
      }).join("");
    }

    init().catch(err => {
      document.body.innerHTML = `<pre style="color:white;padding:24px;white-space:pre-wrap">${err.stack || err}</pre>`;
    });
  </script>
</body>
</html>
""".replace("__INLINE_METADATA_JSON__", inline_metadata_json).replace("__INLINE_FRAMES_JSON__", inline_frames_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--seg-model", type=str, default="nvidia/segformer-b0-finetuned-cityscapes-768-768")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--road-bev-stride", type=int, default=5)
    parser.add_argument("--footprint-scale", type=float, default=0.65)
    parser.add_argument("--grid-smooth-alpha", type=float, default=0.20)
    parser.add_argument("--rgb-quality", type=int, default=88)
    parser.add_argument("--map-scale", type=int, default=3, help="Upscale BEV maps for nicer display.")
    parser.add_argument("--demo-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
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

    depth_cache = DEPTH_CACHE_ROOT / args.date / f"drive_{args.drive}" / "da2_metric_vkitti_vits"
    seg_cache = SEG_CACHE_ROOT / args.date / f"drive_{args.drive}" / safe_model_name(args.seg_model)
    object_path = OBJECT_ROOT / f"road_gated_objects_da2_{args.date}_drive_{args.drive}.jsonl"

    if not depth_cache.exists():
        raise FileNotFoundError(
            f"DA2 depth cache not found: {depth_cache}\n"
            "Run scripts/19_road_gated_object_bev_risk.py first."
        )

    if not seg_cache.exists():
        raise FileNotFoundError(
            f"Road segmentation cache not found: {seg_cache}\n"
            "Run scripts/19_road_gated_object_bev_risk.py first."
        )

    objects_by_frame = load_object_jsonl(object_path)

    demo_name = args.demo_name or f"kitti_{args.date}_drive_{args.drive}"
    demo_dir = DEMO_ROOT / demo_name

    if demo_dir.exists() and args.overwrite:
        shutil.rmtree(demo_dir)

    ensure_dir(demo_dir)

    dirs = {
        "rgb": ensure_dir(demo_dir / "data" / "rgb"),
        "depth": ensure_dir(demo_dir / "data" / "depth"),
        "road_overlay": ensure_dir(demo_dir / "data" / "road_overlay"),
        "road_bev": ensure_dir(demo_dir / "data" / "road_bev"),
        "object_occ": ensure_dir(demo_dir / "data" / "object_occ"),
        "risk": ensure_dir(demo_dir / "data" / "risk"),
        "frame_metadata": ensure_dir(demo_dir / "data" / "frame_metadata"),
    }

    fx, fy, cx, cy = read_kitti_intrinsics(calib_path)
    gh, gw = grid_shape()

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    img_h, img_w = first.shape[:2]
    prev_obj_occ = None

    print("Exporting static HTML demo")
    print("Sequence:", seq_root)
    print("Frames:", n)
    print("Demo dir:", demo_dir)
    print("Objects:", object_path)

    exported_frames = []
    all_frame_metadata = {}

    for frame_idx in tqdm(range(n)):
        frame_id = f"{frame_idx:06d}"

        rgb = cv2.imread(str(image_paths[frame_idx]))
        if rgb is None:
            continue

        depth_path = depth_cache / f"{frame_idx:010d}.npy"
        seg_path = seg_cache / f"{frame_idx:010d}.npy"

        if not depth_path.exists():
            print(f"Skipping frame {frame_idx}: missing depth {depth_path}")
            continue

        if not seg_path.exists():
            print(f"Skipping frame {frame_idx}: missing segmentation {seg_path}")
            continue

        depth = np.load(depth_path).astype(np.float32)
        label_mask = np.load(seg_path).astype(np.uint8)

        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        if label_mask.shape[:2] != rgb.shape[:2]:
            label_mask = cv2.resize(label_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        rec = objects_by_frame.get(frame_idx, {"objects": [], "reject_counts": {}})
        objects = sanitize_objects_for_html(rec.get("objects", []))
        reject_counts = rec.get("reject_counts", {})

        road_bev = road_mask_to_bev(
            depth=depth,
            label_mask=label_mask,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            stride=args.road_bev_stride,
        )

        obj_occ = np.zeros((gh, gw), dtype=np.float32)
        for obj in objects:
            draw_rect_object_occupancy(obj_occ, obj, footprint_scale=args.footprint_scale)

        obj_occ = cv2.GaussianBlur(obj_occ, (3, 3), 0)
        obj_occ = np.clip(obj_occ, 0.0, 1.0)

        if prev_obj_occ is not None and args.grid_smooth_alpha > 0:
            obj_occ = (
                args.grid_smooth_alpha * prev_obj_occ
                + (1.0 - args.grid_smooth_alpha) * obj_occ
            )
            obj_occ = np.clip(obj_occ, 0.0, 1.0)

        prev_obj_occ = obj_occ.copy()
        risk = object_occ_to_risk(obj_occ)

        depth_color = colorize_depth(depth, max_depth=args.max_depth)
        road_overlay = overlay_road_sidewalk(rgb, label_mask)
        road_bev_color = make_road_bev_color(road_bev)
        obj_occ_color = make_object_occ_color(obj_occ)
        risk_color = colorize_map(risk, cv2.COLORMAP_JET)

        if args.map_scale > 1:
            road_bev_color = cv2.resize(
                road_bev_color,
                (road_bev_color.shape[1] * args.map_scale, road_bev_color.shape[0] * args.map_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            obj_occ_color = cv2.resize(
                obj_occ_color,
                (obj_occ_color.shape[1] * args.map_scale, obj_occ_color.shape[0] * args.map_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            risk_color = cv2.resize(
                risk_color,
                (risk_color.shape[1] * args.map_scale, risk_color.shape[0] * args.map_scale),
                interpolation=cv2.INTER_NEAREST,
            )

        write_img(dirs["rgb"] / f"{frame_id}.jpg", rgb, jpg_quality=args.rgb_quality)
        write_img(dirs["depth"] / f"{frame_id}.jpg", depth_color, jpg_quality=90)
        write_img(dirs["road_overlay"] / f"{frame_id}.jpg", road_overlay, jpg_quality=90)
        write_img(dirs["road_bev"] / f"{frame_id}.png", road_bev_color)
        write_img(dirs["object_occ"] / f"{frame_id}.png", obj_occ_color)
        write_img(dirs["risk"] / f"{frame_id}.png", risk_color)

        frame_json = {
            "frame_idx": frame_idx,
            "frame_id": frame_id,
            "timestamp_sec": frame_idx / 10.0,
            "image_size": {
                "width": int(img_w),
                "height": int(img_h),
            },
            "objects": objects,
            "reject_counts": reject_counts,
            "assets": {
                "rgb": f"data/rgb/{frame_id}.jpg",
                "depth": f"data/depth/{frame_id}.jpg",
                "road_overlay": f"data/road_overlay/{frame_id}.jpg",
                "road_bev": f"data/road_bev/{frame_id}.png",
                "object_occ": f"data/object_occ/{frame_id}.png",
                "risk": f"data/risk/{frame_id}.png",
            },
        }

        with (dirs["frame_metadata"] / f"{frame_id}.json").open("w") as f:
            json.dump(frame_json, f, indent=2)

        all_frame_metadata[str(frame_idx)] = frame_json
        exported_frames.append(frame_idx)

    metadata = {
        "title": "LiftDepth Risk Demo",
        "date": args.date,
        "drive": args.drive,
        "sequence": f"{args.date}_drive_{args.drive}_sync",
        "num_frames": len(exported_frames),
        "fps": 10,
        "image_size": {
            "width": int(img_w),
            "height": int(img_h),
        },
        "bev_size": {
            "width": int(gw),
            "height": int(gh),
            "render_scale": int(args.map_scale),
        },
        "bev_bounds": {
            "forward_min_m": FORWARD_MIN,
            "forward_max_m": FORWARD_MAX,
            "side_min_m": SIDE_MIN,
            "side_max_m": SIDE_MAX,
            "resolution_m_per_cell": BEV_RES,
        },
        "pipeline": [
            "RGB frame",
            "DA2 metric depth",
            "Road/sidewalk segmentation",
            "YOLO tracking",
            "Road-gated object projection",
            "Object occupancy grid",
            "Object-aware risk map",
        ],
        "notes": {
            "rgb_boxes": "Drawn dynamically by HTML canvas from JSON metadata.",
            "bev_labels": "Drawn dynamically by HTML canvas from JSON metadata.",
            "road_bev": "Projected road segmentation using DA2 depth and KITTI intrinsics.",
            "object_occ": "Rectangular BEV footprints from road-gated tracked objects.",
        },
        "frames": [f"{i:06d}" for i in exported_frames],
    }

    with (demo_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    inline_metadata_json = json.dumps(metadata)
    inline_frames_json = json.dumps(all_frame_metadata)
    (demo_dir / "index.html").write_text(
        build_index_html(inline_metadata_json, inline_frames_json),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print("Open:")
    print(f"  xdg-open {demo_dir / 'index.html'}")
    print()
    print("This version embeds JSON in index.html, so file:// should work.")
    print("If images still do not load for any browser-specific reason, run:")
    print(f"  cd {demo_dir} && python -m http.server 8000")
    print("  then open http://localhost:8000")


if __name__ == "__main__":
    main()
