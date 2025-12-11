"""
Lightweight end-to-end script for running the stereo sorting pipeline.

Edit LEFT_IMAGE and RIGHT_IMAGE below (or pass new Paths into run_pipeline)
and run:

    python pipeline.py

It will print the derived sorting rule and the list of (x, y, z, bin) tuples.
"""

from pathlib import Path

from stereo_sort import StereoSort

# Edit these to point to your stereo tensors.
LEFT_IMAGE = Path("tensors/sample_00000_left.pt")
RIGHT_IMAGE = Path("tensors/sample_00000_right.pt")


def run_pipeline(left_path: Path, right_path: Path, min_area: float = 200.0) -> None:
    sorter = StereoSort()
    detections = sorter.summarize_scene_stereo(
        left_path,
        right_path,
        min_area=min_area,
    )
    if not detections:
        print("No stereo detections; cannot derive rule.")
        return

    attribute, tuples_xyzb, _ = sorter.apply_sort_rule(detections)
    if attribute is None or not tuples_xyzb:
        print("Unable to determine sorting rule.")
        return

    print(f"Sorting rule: {attribute}")
    print("Tuples:")
    for idx, (x, y, z, bin_idx) in enumerate(tuples_xyzb, 1):
        print(f"  {idx}. ({x:.4f}, {y:.4f}, {z:.4f}, {bin_idx})")


if __name__ == "__main__":
    run_pipeline(LEFT_IMAGE, RIGHT_IMAGE)
