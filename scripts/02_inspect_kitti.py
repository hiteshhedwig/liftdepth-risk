from pathlib import Path
import argparse


def inspect_sequence(date: str, drive: str):
    data_root = Path("data/kitti/raw")
    seq_root = data_root / date / f"{date}_drive_{drive}_sync"

    image_02 = seq_root / "image_02" / "data"
    image_03 = seq_root / "image_03" / "data"
    velo = seq_root / "velodyne_points" / "data"
    oxts = seq_root / "oxts" / "data"

    print("=" * 80)
    print(f"Date : {date}")
    print(f"Drive: {drive}")
    print("Sequence root:", seq_root)
    print("Exists:", seq_root.exists())

    print("\nCounts:")
    print("Left color images image_02:", len(list(image_02.glob("*.png"))))
    print("Right color images image_03:", len(list(image_03.glob("*.png"))))
    print("Velodyne scans:", len(list(velo.glob("*.bin"))))
    print("OXTS files:", len(list(oxts.glob("*.txt"))))

    print("\nFirst 5 left camera images:")
    for p in sorted(image_02.glob("*.png"))[:5]:
        print(p)

    print("\nCalibration files:")
    for p in sorted((data_root / date).glob("calib_*.txt")):
        print(p)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2011_09_26")
    parser.add_argument("--drive", type=str, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all downloaded drives for the given date.",
    )

    args = parser.parse_args()

    data_root = Path("data/kitti/raw")

    if args.all:
        date_root = data_root / args.date
        seqs = sorted(date_root.glob(f"{args.date}_drive_*_sync"))

        if not seqs:
            print("No sequences found in:", date_root)
            return

        for seq in seqs:
            # Example folder:
            # 2011_09_26_drive_0005_sync
            drive = seq.name.split("_drive_")[1].split("_sync")[0]
            inspect_sequence(args.date, drive)

    else:
        drive = args.drive if args.drive is not None else "0001"
        inspect_sequence(args.date, drive)


if __name__ == "__main__":
    main()
