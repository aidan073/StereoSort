"""
CLI harness for running StereoSort contour classification workflows.

Usage:
    python testing.py --scene /path/to/frame.pt
        Summarize detections (color, shape, centroid) for a single tensor.

    python testing.py --dir /path/to/folder
        Batch a directory of tensors; prints detections and saves annotations beside data.

    python testing.py --dataset StereoSort/dataset.json --out predictions.json
        Run contour classification for every sample from dataset.json and write detections
        to predictions.json (defaults to dataset_predictions.json if --out omitted).

    python testing.py --compare --dataset StereoSort/dataset.json --predictions predictions.json
        Compare saved predictions against the dataset ground truth and print accuracy stats.

Only one of (--scene, --dir, --dataset, --compare) should be active at a time.
"""

import argparse
from pathlib import Path

from StereoSort.stereo_sort import StereoSort


def main():
    parser = argparse.ArgumentParser(
        description="Quick harness for running StereoSort contour classification."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        help="Path to a single .pt tensor to summarize (prints detections).",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory of .pt tensors to batch process.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset.json for running the evaluation pipeline.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to predictions json (use with --compare).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare dataset.json against predictions.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path for saving predictions json (only with --dataset).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=200.0,
        help="Minimum contour area (in pixels) to keep.",
    )
    args = parser.parse_args()

    sorter = StereoSort()

    if args.compare:
        if not args.dataset or not args.predictions:
            parser.error("--compare requires both --dataset and --predictions")
        sorter.compare_predictions(args.dataset, args.predictions)
    elif args.dataset:
        sorter.classify_dataset(
            args.dataset,
            output_json=args.out,
            min_area=args.min_area,
        )
    elif args.scene:
        sorter.summarize_scene(args.scene, min_area=args.min_area)
    elif args.dir:
        sorter.classify_shapes_in_directory(args.dir, min_area=args.min_area)
    else:
        parser.error("Provide --dataset, --scene, or --dir.")


if __name__ == "__main__":
    main()
