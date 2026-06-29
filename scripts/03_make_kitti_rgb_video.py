from pathlib import Path
import argparse
import cv2
from tqdm import tqdm


def make_rgb_video(date: str, drive: str, fps: int = 10):
    data_root = Path("data/kitti/raw")
    seq_root = data_root / date / f"{date}_drive_{drive}_sync"
    image_dir = seq_root / "image_02" / "data"

    output_dir = Path("outputs/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"rgb_preview_{date}_drive_{drive}.mp4"

    image_paths = sorted(image_dir.glob("*.png"))

    print("Sequence:", seq_root)
    print("Image directory:", image_dir)
    print("Images found:", len(image_paths))

    if not image_paths:
        raise RuntimeError(f"No images found at: {image_dir}")

    first = cv2.imread(str(image_paths[0]))

    if first is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for idx, img_path in enumerate(tqdm(image_paths)):
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        cv2.putText(
            frame,
            f"KITTI {date} drive {drive} | frame {idx:04d}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()

    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)

    args = parser.parse_args()

    make_rgb_video(args.date, args.drive, args.fps)


if __name__ == "__main__":
    main()
