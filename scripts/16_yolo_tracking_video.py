from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


DATA_ROOT = Path("data/kitti/raw")
OUTPUT_VIDEO_DIR = Path("outputs/videos")
OUTPUT_TRACK_DIR = Path("outputs/tracks/yolo")

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TRACK_DIR.mkdir(parents=True, exist_ok=True)


# COCO class ids:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7]


def draw_label(img, x1, y1, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x1 = int(x1)
    y1 = int(y1)

    y_text = max(y1 - 8, th + 8)

    cv2.rectangle(
        img,
        (x1, y_text - th - baseline - 4),
        (x1 + tw + 6, y_text + baseline),
        color,
        -1,
    )

    cv2.putText(
        img,
        text,
        (x1 + 3, y_text - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def color_for_track(track_id):
    if track_id is None:
        return (80, 80, 255)

    rng = np.random.default_rng(int(track_id) + 12345)
    color = rng.integers(80, 255, size=3).tolist()
    return tuple(int(c) for c in color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=DEFAULT_CLASSES,
        help="COCO class IDs to track. Default: person bicycle car motorcycle bus truck.",
    )
    args = parser.parse_args()

    seq_root = DATA_ROOT / args.date / f"{args.date}_drive_{args.drive}_sync"
    image_dir = seq_root / "image_02" / "data"

    image_paths = sorted(image_dir.glob("*.png"))

    print("Sequence:", seq_root)
    print("Images found:", len(image_paths))
    print("Model:", args.model)
    print("Tracker:", args.tracker)
    print("Classes:", args.classes)

    if not image_paths:
        raise RuntimeError(f"No images found at: {image_dir}")

    n = len(image_paths)
    if args.max_frames is not None:
        n = min(n, args.max_frames)

    out_video = OUTPUT_VIDEO_DIR / f"phase9a_yolo_tracking_{args.date}_drive_{args.drive}.mp4"
    out_jsonl = OUTPUT_TRACK_DIR / f"tracks_{args.date}_drive_{args.drive}.jsonl"

    model = YOLO(args.model)

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h),
    )

    with out_jsonl.open("w") as f:
        for frame_idx in tqdm(range(n)):
            img_path = image_paths[frame_idx]
            frame = cv2.imread(str(img_path))

            if frame is None:
                continue

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                classes=args.classes,
                verbose=False,
            )

            frame_records = []

            if results and len(results) > 0:
                result = results[0]
                names = result.names

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()

                    if boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        ids = np.array([-1] * len(xyxy), dtype=int)

                    for box, c, score, tid in zip(xyxy, cls, conf, ids):
                        x1, y1, x2, y2 = box.tolist()

                        class_name = names.get(int(c), str(c))
                        track_id = None if int(tid) < 0 else int(tid)

                        color = color_for_track(track_id)

                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            2,
                        )

                        label = (
                            f"ID {track_id} | {class_name} {score:.2f}"
                            if track_id is not None
                            else f"{class_name} {score:.2f}"
                        )

                        draw_label(frame, x1, y1, label, color)

                        frame_records.append(
                            {
                                "frame_idx": frame_idx,
                                "image_path": str(img_path),
                                "track_id": track_id,
                                "class_id": int(c),
                                "class_name": class_name,
                                "confidence": float(score),
                                "bbox_xyxy": [
                                    float(x1),
                                    float(y1),
                                    float(x2),
                                    float(y2),
                                ],
                            }
                        )

            cv2.putText(
                frame,
                f"KITTI {args.date} drive {args.drive} | YOLO tracking | frame {frame_idx:04d}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

            f.write(
                json.dumps(
                    {
                        "frame_idx": frame_idx,
                        "image_path": str(img_path),
                        "objects": frame_records,
                    }
                )
                + "\n"
            )

    writer.release()

    print("Saved video:", out_video)
    print("Saved tracks:", out_jsonl)


if __name__ == "__main__":
    main()
