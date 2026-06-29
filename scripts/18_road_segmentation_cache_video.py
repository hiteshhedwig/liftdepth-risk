from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


DATA_ROOT = Path("data/kitti/raw")
OUTPUT_VIDEO_DIR = Path("outputs/videos")
SEG_CACHE_ROOT = Path("outputs/seg_cache/road_seg")

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
SEG_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def safe_model_name(model_name: str):
    return model_name.replace("/", "_").replace("-", "_")


def get_cityscapes_label_ids(config):
    """
    Cityscapes-style labels usually include:
      road, sidewalk, building, wall, fence, pole, traffic light, ...
    We resolve IDs from model.config.id2label instead of hardcoding.
    """
    id2label = config.id2label

    road_ids = []
    sidewalk_ids = []

    for k, v in id2label.items():
        idx = int(k)
        name = str(v).lower().strip()

        if name == "road":
            road_ids.append(idx)

        if name == "sidewalk":
            sidewalk_ids.append(idx)

    if not road_ids:
        raise RuntimeError(f"Could not find 'road' label in id2label: {id2label}")

    print("Road label IDs:", road_ids)
    print("Sidewalk label IDs:", sidewalk_ids)

    return road_ids, sidewalk_ids


def run_segmentation(model, processor, img_path: Path, device, road_ids, sidewalk_ids):
    rgb_pil = Image.open(img_path).convert("RGB")
    w, h = rgb_pil.size

    inputs = processor(images=rgb_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    logits = F.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    road_mask = np.isin(pred, road_ids)
    sidewalk_mask = np.isin(pred, sidewalk_ids)

    # 0 background, 1 road, 2 sidewalk
    label_mask = np.zeros_like(pred, dtype=np.uint8)
    label_mask[road_mask] = 1
    label_mask[sidewalk_mask] = 2

    return label_mask


def get_masks(label_mask: np.ndarray, dilate_px: int):
    road = (label_mask == 1).astype(np.uint8)
    sidewalk = (label_mask == 2).astype(np.uint8)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        road_dilated = cv2.dilate(road, kernel, iterations=1)
    else:
        road_dilated = road.copy()

    return road, sidewalk, road_dilated


def make_color_mask(label_mask: np.ndarray):
    h, w = label_mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    # BGR colors
    color[label_mask == 1] = (70, 220, 70)      # road
    color[label_mask == 2] = (220, 180, 70)     # sidewalk

    return color


def make_binary_color(mask: np.ndarray):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = (70, 220, 70)
    return out


def overlay_mask(rgb_bgr: np.ndarray, label_mask: np.ndarray, alpha: float = 0.45):
    color = make_color_mask(label_mask)
    mask = label_mask > 0

    out = rgb_bgr.copy()
    out[mask] = (
        (1.0 - alpha) * out[mask].astype(np.float32)
        + alpha * color[mask].astype(np.float32)
    ).astype(np.uint8)

    return out


def make_panel(rgb_bgr, label_mask, road_dilated, frame_idx, date, drive):
    top_tile_w = 621
    top_tile_h = 188

    bottom_tile_w = 621
    bottom_tile_h = 376

    overlay = overlay_mask(rgb_bgr, label_mask)
    mask_color = make_color_mask(label_mask)
    road_dilated_color = make_binary_color(road_dilated)

    rgb_tile = cv2.resize(rgb_bgr, (top_tile_w, top_tile_h))
    overlay_tile = cv2.resize(overlay, (top_tile_w, top_tile_h))

    mask_tile = cv2.resize(mask_color, (bottom_tile_w, bottom_tile_h), interpolation=cv2.INTER_NEAREST)
    road_tile = cv2.resize(road_dilated_color, (bottom_tile_w, bottom_tile_h), interpolation=cv2.INTER_NEAREST)

    top = np.concatenate([rgb_tile, overlay_tile], axis=1)
    bottom = np.concatenate([mask_tile, road_tile], axis=1)
    panel = np.concatenate([top, bottom], axis=0)

    cv2.putText(panel, f"RGB frame {frame_idx:04d}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Road/sidewalk overlay", (top_tile_w + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Segmentation mask: road + sidewalk", (15, top_tile_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, "Dilated road gate for vehicles", (top_tile_w + 15, top_tile_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(panel, f"KITTI {date} drive {drive}", (15, top_tile_h + bottom_tile_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-cityscapes-768-768")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dilate-road-px", type=int, default=12)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    seq_root = DATA_ROOT / args.date / f"{args.date}_drive_{args.drive}_sync"
    image_dir = seq_root / "image_02" / "data"
    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise RuntimeError(f"No images found at: {image_dir}")

    n = len(image_paths)
    if args.max_frames is not None:
        n = min(n, args.max_frames)

    model_key = safe_model_name(args.model)
    cache_dir = SEG_CACHE_ROOT / args.date / f"drive_{args.drive}" / model_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_video = OUTPUT_VIDEO_DIR / f"phase10a_road_seg_{args.date}_drive_{args.drive}_{model_key}.mp4"

    print("Sequence:", seq_root)
    print("Images:", len(image_paths))
    print("Frames used:", n)
    print("Model:", args.model)
    print("Cache:", cache_dir)
    print("Output:", out_video)
    print("Road dilation px:", args.dilate_road_px)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading segmentation model...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model, use_safetensors=True).to(device).eval()

    road_ids, sidewalk_ids = get_cityscapes_label_ids(model.config)

    panel_w = 621 * 2
    panel_h = 188 + 376

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (panel_w, panel_h),
    )

    for frame_idx in tqdm(range(n)):
        img_path = image_paths[frame_idx]
        rgb_bgr = cv2.imread(str(img_path))

        if rgb_bgr is None:
            continue

        cache_path = cache_dir / f"{frame_idx:010d}.npy"

        if cache_path.exists() and not args.overwrite_cache:
            label_mask = np.load(cache_path).astype(np.uint8)
        else:
            label_mask = run_segmentation(
                model=model,
                processor=processor,
                img_path=img_path,
                device=device,
                road_ids=road_ids,
                sidewalk_ids=sidewalk_ids,
            )
            np.save(cache_path, label_mask)

        _, _, road_dilated = get_masks(label_mask, dilate_px=args.dilate_road_px)

        panel = make_panel(
            rgb_bgr=rgb_bgr,
            label_mask=label_mask,
            road_dilated=road_dilated,
            frame_idx=frame_idx,
            date=args.date,
            drive=args.drive,
        )

        writer.write(panel)

    writer.release()

    print("Saved video:", out_video)
    print("Saved segmentation cache:", cache_dir)


if __name__ == "__main__":
    main()
