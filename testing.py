"""
CLI harness for running StereoSort contour classification workflows.

Usage:
    python testing.py --scene /path/to/frame.pt
        Summarize detections (color, shape, centroid) for a single tensor.

    python testing.py --stereo-left left.pt --stereo-right right.pt
        Summarize detections for a stereo pair, derive world coordinates from
        the centroids, and print the variance-based bin assignments.

    python testing.py --dir /path/to/folder
        Batch a directory of tensors; prints detections and saves annotations beside data.

    python testing.py --dataset dataset.json --out predictions.json
        Run contour classification for every sample from dataset.json and write detections
        to predictions.json (defaults to dataset_predictions.json if --out omitted).

    python testing.py --compare --dataset dataset.json --predictions predictions.json
        Compare saved predictions against the dataset ground truth and print accuracy stats.

Only one of (--scene, --stereo-*, --dir, --dataset, --compare) should be active at a time.
"""

import argparse
from pathlib import Path

from stereo_sort import StereoSort


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
        "--stereo-left",
        type=Path,
        help="Path to left .pt tensor for stereo summary.",
    )
    parser.add_argument(
        "--stereo-right",
        type=Path,
        help="Path to right .pt tensor for stereo summary.",
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
    elif args.stereo_left or args.stereo_right:
        if not (args.stereo_left and args.stereo_right):
            parser.error("--stereo-left and --stereo-right must be provided together")
        stereo_matches = sorter.summarize_scene_stereo(
            args.stereo_left,
            args.stereo_right,
            min_area=args.min_area,
        )
        if stereo_matches:
            attribute, binned, sorted_dets = sorter.apply_sort_rule(stereo_matches)
            if attribute is None:
                print("Unable to determine sorting rule from stereo detections.")
            else:
                unique_bins = {bin_idx for *_, bin_idx in binned}
                print(f"\nSorting rule: {attribute} (bins={len(unique_bins)})")
                for idx, (x_world, y_world, z_world, bin_idx) in enumerate(binned, 1):
                    det = sorted_dets[idx - 1]
                    attr_label = det.get(attribute)
                    color = det.get("color")
                    shape = det.get("shape")
                    print(
                        f"  {idx}. bin={bin_idx} {attribute}={attr_label} "
                        f"(color={color}, shape={shape}) X={x_world:.4f} "
                        f"Y={y_world:.4f} Z={z_world:.4f}"
                    )
    elif args.scene:
        sorter.summarize_scene(args.scene, min_area=args.min_area)
    elif args.dir:
        sorter.classify_shapes_in_directory(args.dir, min_area=args.min_area)
    else:
        parser.error("Provide --dataset, --scene, --stereo-*, or --dir.")


if __name__ == "__main__":
    main()
