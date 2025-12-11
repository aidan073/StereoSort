"""
This file holds functions that allow going from stereo images to ordered target
positions, based on some sorting criteria.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2 as cv
import numpy as np
import torch


class StereoSort:
    _KERNEL = np.ones((5, 5), dtype=np.uint8)
    _COLOR_THRESHOLDS = {
        "red": [
            (np.array([0, 180, 180]), np.array([10, 255, 255])),
            (np.array([170, 180, 180]), np.array([179, 255, 255])),
        ],
        "green": [(np.array([50, 180, 180]), np.array([70, 255, 255]))],
        "blue": [(np.array([110, 180, 180]), np.array([130, 255, 255]))],
    }

    def __init__(self, data_bundle: dict = None):
        """
        data_bundle (dict): All the parameters necessary for the StereoSort methods. You can set this now,
            or set it before the first call to any other StereoSort method using StereoSort.load_data_bundle()
            bundle entries example:
                'resolution' = tuple(3840, 2160)
                'center_of_image' = tuple(1920, 1080)
                'center_of_circle' = tuple(0, 0)
                'distance_from_camera' = 0.8616
                'baseline' = 0.11988
                'fx' = 1472
                'fy' = 1472
                inner_radius = 0.20
                outer_radius = 0.35

            bundle entries explanations:
                resolution (pixels): A tuple containing the cameras resolution (both cameras must have same resolution)
                center_of_image (pixels): A tuple containing the center pixel of the image (both cameras must have same center).
                center_of_circle (meters): A tuple containing the coords of the circle relative to the env origin.
                distance_from_camera (meters): A float value describing the distance of the circle's plane from the camera.
                baseline (meters): The baseline of the stereo camera.
                fx: Effective focal length in x-pixel axis
                fy: Effective focal length in y-pixel axis
                inner_radius (meters): The inner radius that the pickables are allowed to spawn past
                outer_radius (meters): The outer radius that the pickables are allowed to spawn before
        """
        self.data_bundle = None
        self.left_img = None
        self.right_img = None
        self.disparity_map = None
        self._L_mask = None
        self._R_mask = None
        self._output_dir = None

    def load_data_bundle(self, data_bundle: dict):
        self.data_bundle = data_bundle

    def load_images(self, left_img: torch.Tensor, right_img: torch.Tensor) -> None:
        """
        Load stereo images that will be used for feature extraction.

        Args:
            left_img (Tensor): The left image from isaacsim
            right_img (Tensor): The right image from isaacsim
        """
        self.left_img = left_img
        self.right_img = right_img

    def generate_circle_masks(self):
        """
        NOTE: This method was intended for a custom stereo pixel matching algorithm
        in the 'generated_disparity_map' method. I ended up not implementing that
        method, but I am keeping this method in here since it may come in use.

        Generates binary annulus masks for left and right cameras.
        The annulus is defined in world coordinates by:
        - center of circle (x, y)
        - inner and outer radii (world units)

        Assumes camera is facing directly downward with no rotation.
        Converts world radii into pixel radii using fx, fy and depth Z.
        """

        if self.data_bundle is None:
            print(
                "[WARNING] StereoSort object did not have data_bundle set, and therefore couldn't generate circle mask."
            )
            return

        # === Unpack data ===
        x_world, y_world = self.data_bundle["center_of_circle"]
        inner_r_world = self.data_bundle["inner_radius"]
        outer_r_world = self.data_bundle["outer_radius"]
        baseline = self.data_bundle["baseline"]
        fx = self.data_bundle["fx"]
        fy = self.data_bundle["fy"]
        Z = self.data_bundle["distance_from_camera"]
        W, H = self.data_bundle["resolution"]  # (width, height)

        # Camera centers in world coordinates (assuming stereo rectified, facing down)
        left_cam_x = -baseline / 2
        right_cam_x = baseline / 2

        # === Project center of annulus into each camera ===
        # Pixel coords: u = fx * ((X_world - cam_center_x) / Z)
        #                v = fy * (Y_world / Z)

        def project_center(cam_x):
            u = fx * ((x_world - cam_x) / Z)
            v = fy * (y_world / Z)
            return u, v

        left_cx, left_cy = project_center(left_cam_x)
        right_cx, right_cy = project_center(right_cam_x)

        # === Convert world radii to pixel radii ===
        # Radius is scalar → r_px = fx * (r_world / Z)
        # (We use fx; since camera looks down, radial expansion primarily maps in x-pixels)
        inner_r_px = fx * (inner_r_world / Z)
        outer_r_px = fx * (outer_r_world / Z)

        # === Build coordinate grid ===
        ys = torch.arange(H, device="cpu").view(-1, 1).float()
        xs = torch.arange(W, device="cpu").view(1, -1).float()

        # === Build masks for left and right cameras ===
        def make_mask(cx, cy):
            dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
            return (dist2 >= inner_r_px**2) & (dist2 <= outer_r_px**2)

        L_mask = make_mask(left_cx, left_cy)
        R_mask = make_mask(right_cx, right_cy)

        # Store results
        self._L_mask = L_mask
        self._R_mask = R_mask

    def generate_disparity_map(self):
        """
        One of the features we want to extract in this class is shape. A disparity
        map will be needed for this, because we have to isolate the top face
        of each 3D object to determine its shape. We can find the top face of each
        pickable by isolating the highest disparity face of each pickable.

        This algorithm is extremely simplistic, and requires some *assumptions*:
        1. Background is fairly gray scaled (slight lenience allowed)
        2. Cameras are perfectly rectified (no lenience)
        3. No other red, green, or blue objects are viewable (circle mask helps with this)
        4. The cameras mostly see the same sides of the pickables (slight lenience allowed)
        5. Lighting is not hitting multiple sides of the pickables with the same intensity
        6. Cameras face directly down at the table surface (TODO: Fix this with coord transform)

        Disparity is simply the pixel distance between two corresponding pixels in a
        stereo image pair. This algorithm finds a corresponding pixel from a reference
        pixel in the left image by scanning the right image starting from the left most
        pixel of the same row, and scanning until a pixel has a matching rgb value (within some
        threshold), and hasn't been claimed as a pair by another pixel yet.
        """

    def centroids_to_features(points: List[Tuple]):
        """
        Take a list of pickable centroids,
        """

    def ncc_horizontal(
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        template_center: Tuple,
        template_radius: int,
        max_disp: int,
        color_hack_threshold: float = None,
    ):
        """
        Compute normalized cross-correlation (NCC) along a single row,
        scanning only horizontally within [-max_disp, max_disp].
        Takes a template from the right image, and scans from the start of the
        same row in the left image.

        Args:
            left_img: H x W grayscale left image
            right_img: H x W grayscale right image
            template_center: (x, y) tuple in right image
            template_radius: half-size of square template (template size = 2*radius+1)
            max_disp: maximum horizontal distance to search (disparity)
            color_hack_threshold: If set, the rgb euclidean distance between the template_center pixel
                and a left image pixel must be less than the threshold to be considered a match

        Returns:
            best_x: x-coordinate of best match in left image
            best_disp: disparity = x_left - x_right
            max_ncc: maximum NCC score
        """
        H, W = left_img.shape
        x, y = template_center
        ts = template_radius

        # Ensure template fits inside right image
        if x - ts < 0 or x + ts >= W or y - ts < 0 or y + ts >= H:
            raise ValueError(
                f"Template of radius: {ts} will exceed border of right image."
            )

        # Extract template
        template = right_img[y - ts : y + ts + 1, x - ts : x + ts + 1].astype(
            torch.float32
        )

        t_mean = torch.mean(template)
        t_std = torch.std(template)
        t_std = max(t_std, 1e-6)

        best_ncc = -1
        best_x = x

        tw = th = 2 * ts + 1
        row_start = y - ts
        row_end = y + ts + 1

        # Ensure vertical patch slice fits inside left image
        if row_start < 0 or row_end > H:
            raise ValueError("Template window exceeds vertical bounds in left image")

        # Horizontal scan limits
        x_start = max(0, x - max_disp)
        x_end = min(W - tw, x + max_disp)

        for xr in range(x_start, x_end + 1):
            patch = left_img[row_start:row_end, xr : xr + tw].astype(torch.float32)

            p_mean = torch.mean(patch)
            p_std = max(torch.std(patch), 1e-6)
            ncc = torch.sum((template - t_mean) * (patch - p_mean)) / (
                t_std * p_std * tw * th
            )

            if color_hack_threshold is not None:
                template_pixel = right_img[y, x].float()
                patch_pixel = left_img[y, xr].float()
                if torch.norm(template_pixel - patch_pixel) >= color_hack_threshold:
                    continue

            if ncc > best_ncc:
                best_ncc = ncc
                best_x = xr

        best_disp = best_x - x
        return best_x, best_disp, best_ncc

    @staticmethod
    def _load_pt_image(path: Path) -> np.ndarray:
        """
        Load a tensor saved via torch.save and convert it to an OpenCV BGR image.
        """
        tensor = torch.load(path, map_location="cpu")

        if not torch.is_tensor(tensor):
            raise ValueError(f"{path.name} does not contain a tensor.")

        if not torch.is_floating_point(tensor):
            tensor = tensor.float()

        tensor = tensor.squeeze()

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # grayscale
        elif tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)  # channels-last -> channels-first

        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)

        if tensor.max() > 1:
            tensor = tensor / 255.0

        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte().cpu().numpy()
        rgb = np.transpose(tensor, (1, 2, 0))
        return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    @staticmethod
    def _classify_shape(contour: np.ndarray, min_area: float = 200.0) -> str:
        area = cv.contourArea(contour)
        if area < min_area:
            return None
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        vertices = len(approx)
        circularity = (4 * np.pi * area) / (peri * peri + 1e-5)

        if vertices == 4:
            return "cube"
        if circularity > 0.7:
            return "cylinder"
        if vertices == 3:
            return "triangular_prism"
        return "unknown"

    @staticmethod
    def _classify_color_at_pixel(
        hsv_img: np.ndarray, cx: int, cy: int
    ) -> Tuple[str, Tuple[int, int, int]]:
        hue, sat, val = hsv_img[cy, cx]
        if (hue <= 10 or hue >= 170) and sat > 150 and val > 150:
            return "red", (255, 255, 255)
        if 50 <= hue <= 70 and sat > 150 and val > 150:
            return "green", (255, 0, 255)
        if 110 <= hue <= 130 and sat > 150 and val > 150:
            return "blue", (0, 255, 255)
        return "unknown", (200, 200, 200)

    @classmethod
    def _create_color_mask(cls, hsv_img: np.ndarray) -> np.ndarray:
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

        for color_ranges in cls._COLOR_THRESHOLDS.values():
            color_mask = np.zeros_like(mask)
            for lower, upper in color_ranges:
                color_mask = cv.bitwise_or(color_mask, cv.inRange(hsv_img, lower, upper))
            mask = cv.bitwise_or(mask, color_mask)

        return mask

    def _detect_shapes(
        self,
        img: np.ndarray,
        hsv: np.ndarray,
        file_name: str,
        min_area: float,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run the contour-based detection on a single image.
        Returns both the detections and the annotated visualization.
        """
        mask = self._create_color_mask(hsv)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self._KERNEL)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        out = img.copy()
        detections: List[Dict[str, Any]] = []

        for contour in contours:
            shape = self._classify_shape(contour, min_area)
            if not shape:
                continue

            M = cv.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            color_name, draw_color = self._classify_color_at_pixel(hsv, cx, cy)
            label = f"{color_name} {shape}"

            cv.drawContours(out, [contour], -1, draw_color, 2)
            cv.circle(out, (cx, cy), 4, draw_color, -1)
            cv.putText(
                out,
                label,
                (cx - 60, cy - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                draw_color,
                2,
                cv.LINE_AA,
            )

            pixel_hsv = hsv[cy, cx].tolist()
            detections.append(
                {
                    "file": file_name,
                    "label": label,
                    "shape": shape,
                    "color": color_name,
                    "centroid": (cx, cy),
                    "hsv": pixel_hsv,
                }
            )

        return detections, out

    def classify_shapes_in_directory(
        self,
        input_dir: Path,
        output_dir: Path = None,
        min_area: float = 200.0,
    ) -> List[Dict[str, Any]]:
        """
        Replicates the contouring pipeline over a directory of .pt images.
        Returns the list of detected labels per file, and saves annotated images.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

        output_path = Path(output_dir) if output_dir else input_path / "abscolor_classified"
        output_path.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_path

        results: List[Dict[str, Any]] = []

        for file in sorted(input_path.glob("*.pt")):
            img = self._load_pt_image(file)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            detections, annotated = self._detect_shapes(img, hsv, file.name, min_area)
            for detection in detections:
                cx, cy = detection["centroid"]
                print(
                    f"{file.name}: {detection['label']} centroid=({cx},{cy}) HSV={detection['hsv']}"
                )
            results.extend(detections)

            cv.imwrite(str(output_path / f"{file.stem}.jpg"), annotated)

        print(f"\nColor + shape labeling (high contrast) saved to: {output_path}")
        return results

    def summarize_scene(
        self,
        scene_path: Path,
        min_area: float = 200.0,
    ) -> List[Dict[str, Any]]:
        """
        Pipeline helper for a single scene (.pt tensor).
        Prints detections in a readable format and returns them for further use.
        """
        scene_path = Path(scene_path)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene does not exist: {scene_path}")

        img = self._load_pt_image(scene_path)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        detections, _ = self._detect_shapes(img, hsv, scene_path.name, min_area)

        if not detections:
            print(f"{scene_path.name}: No objects detected.")
            return []

        print(f"{scene_path.name} detections:")
        for idx, detection in enumerate(detections, 1):
            cx, cy = detection["centroid"]
            hsv_values = detection["hsv"]
            print(
                f"  {idx}. color={detection['color']}, shape={detection['shape']}, centroid=({cx},{cy}), HSV={hsv_values}"
            )

        return detections

    def summarize_scene_stereo(
        self,
        left_scene_path: Path,
        right_scene_path: Path,
        min_area: float = 200.0,
    ) -> List[Dict[str, Any]]:
        """
        Process a left/right scene pair and report stereo disparities per object.

        Args:
            left_scene_path: Path to the left camera tensor (.pt).
            right_scene_path: Path to the right camera tensor (.pt).
            min_area: Minimum contour area to be considered a valid object.

        Returns:
            List of dicts with color, shape, left/right centroids, and disparity
            (x_left - x_right) in pixels.
        """

        left_scene_path = Path(left_scene_path)
        right_scene_path = Path(right_scene_path)

        if not left_scene_path.exists():
            raise FileNotFoundError(f"Scene does not exist: {left_scene_path}")
        if not right_scene_path.exists():
            raise FileNotFoundError(f"Scene does not exist: {right_scene_path}")

        left_img = self._load_pt_image(left_scene_path)
        right_img = self._load_pt_image(right_scene_path)

        left_hsv = cv.cvtColor(left_img, cv.COLOR_BGR2HSV)
        right_hsv = cv.cvtColor(right_img, cv.COLOR_BGR2HSV)

        left_detections, _ = self._detect_shapes(
            left_img, left_hsv, left_scene_path.name, min_area
        )
        right_detections, _ = self._detect_shapes(
            right_img, right_hsv, right_scene_path.name, min_area
        )

        if not left_detections or not right_detections:
            print(
                f"{left_scene_path.name}/{right_scene_path.name}: Unable to compute disparities; missing detections."
            )
            return []

        matches: List[Dict[str, Any]] = []
        used_right_indices = set()

        for left_det in left_detections:
            left_label = (left_det["color"], left_det["shape"])
            left_cx, left_cy = left_det["centroid"]

            best_idx = None
            best_dist = float("inf")

            for idx, right_det in enumerate(right_detections):
                if idx in used_right_indices:
                    continue
                if (right_det["color"], right_det["shape"]) != left_label:
                    continue

                right_cx, right_cy = right_det["centroid"]
                dist = np.hypot(left_cx - right_cx, left_cy - right_cy)

                if dist < best_dist:
                    best_idx = idx
                    best_dist = dist

            if best_idx is None:
                continue

            used_right_indices.add(best_idx)
            right_det = right_detections[best_idx]
            right_cx, right_cy = right_det["centroid"]
            disparity = left_cx - right_cx

            matches.append(
                {
                    "color": left_det["color"],
                    "shape": left_det["shape"],
                    "centroid_left": (left_cx, left_cy),
                    "centroid_right": (right_cx, right_cy),
                    "disparity": disparity,
                }
            )

        if not matches:
            print(
                f"{left_scene_path.name}/{right_scene_path.name}: No stereo matches; color/shape labels did not align."
            )
            return []

        print(f"Stereo scene {left_scene_path.name} vs {right_scene_path.name}:")
        for idx, match in enumerate(matches, 1):
            cx_l, cy_l = match["centroid_left"]
            cx_r, cy_r = match["centroid_right"]
            print(
                f"  {idx}. color={match['color']} shape={match['shape']} "
                f"left=({cx_l},{cy_l}) right=({cx_r},{cy_r}) disparity={match['disparity']} px"
            )

        return matches

    @staticmethod
    def determine_sort_rule(
        detections: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        Decide whether to sort by color or shape based on distinct value counts.

        Returns:
            (attribute, ordered_labels) where attribute i(x,y,z,bin), s 'color' or 'shape' and
            ordered_labels preserves first-seen order for stable binning.
        """
        if not detections:
            return None, []

        colors = [det.get("color") for det in detections]
        shapes = [det.get("shape") for det in detections]

        def unique_ordered(values):
            seen = []
            for val in values:
                if val not in seen:
                    seen.append(val)
            return seen

        unique_colors = unique_ordered(colors)
        unique_shapes = unique_ordered(shapes)

        if len(unique_colors) > len(unique_shapes):
            return "color", unique_colors
        if len(unique_shapes) > len(unique_colors):
            return "shape", unique_shapes
        # Tie: default to color for determinism
        return "color", unique_colors

    def apply_sort_rule(
        self, detections: List[Dict[str, Any]]
    ) -> Tuple[str, List[Tuple[float, float, float, int]], List[Dict[str, Any]]]:
        """
        Apply the variance-based sorting rule to stereo detections.

        Returns:
            attribute used ('color' or 'shape'), a list of tuples
            (X_world, Y_world, Z_world, bin_index), and the detections sorted
            by the same bin order.
        """
        attribute, labels = self.determine_sort_rule(detections)
        if attribute is None:
            return None, [], []

        label_to_bin = {label: idx for idx, label in enumerate(labels)}

        records = []
        for det in detections:
            centroid = det.get("centroid_left") or det.get("centroid")
            if centroid is None:
                continue
            x, y = centroid
            disparity = det.get("disparity", 0.0)
            bin_idx = label_to_bin.get(det.get(attribute))
            if bin_idx is None:
                continue
            disparity_tensor = torch.tensor([disparity], dtype=torch.float32)
            depth_tensor = self.disparity_to_depth(disparity_tensor, side="left")

            u_tensor = torch.tensor([x], dtype=torch.float32)
            v_tensor = torch.tensor([y], dtype=torch.float32)
            coords_tensor = self.pixels_to_coords(
                u_tensor, v_tensor, depth_tensor, side="left"
            )

            coords_list = coords_tensor.reshape(-1).tolist()
            if len(coords_list) < 3:
                continue

            world_x, world_y, world_z = coords_list[:3]
            records.append((bin_idx, (world_x, world_y, world_z, bin_idx), det))

        records.sort(key=lambda item: item[0])
        binned = [item[1] for item in records]
        sorted_detections = [item[2] for item in records]
        return attribute, binned, sorted_detections

    def classify_dataset(
        self,
        dataset_json: Path,
        output_json: Path = None,
        min_area: float = 200.0,
    ) -> Path:
        """
        Run contour-based classification over every sample in dataset_json.
        Outputs a JSON file with detections for downstream accuracy evaluation.
        """
        dataset_path = Path(dataset_json)
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset.json not found: {dataset_path}")

        with dataset_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

        samples = dataset.get("samples", [])
        dataset_root = dataset_path.parent

        predictions: Dict[str, Any] = {
            "source_dataset": str(dataset_path),
            "created": datetime.utcnow().isoformat(),
            "min_area": min_area,
            "num_samples": len(samples),
            "samples": [],
        }

        for sample in samples:
            sample_id = sample.get("sample_id")
            left_rel = sample.get("left_tensor_path")
            if not left_rel:
                continue

            left_path = dataset_root / left_rel
            if not left_path.exists():
                print(f"[WARN] Left tensor missing for sample {sample_id}: {left_path}")
                continue

            img = self._load_pt_image(left_path)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            detections, _ = self._detect_shapes(img, hsv, left_path.name, min_area)

            predictions["samples"].append(
                {
                    "sample_id": sample_id,
                    "left_tensor_path": str(left_rel),
                    "detections": detections,
                }
            )

        output_path = (
            Path(output_json)
            if output_json
            else dataset_path.with_name(f"{dataset_path.stem}_predictions.json")
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

        print(f"Predictions saved to: {output_path}")
        return output_path

    def compare_predictions(
        self,
        dataset_json: Path,
        predictions_json: Path,
    ) -> Dict[str, Any]:
        """
        Compare saved predictions against dataset ground truth.
        Returns accuracy metrics (color+shape pairs) and prints a brief summary.
        """
        dataset_path = Path(dataset_json)
        predictions_path = Path(predictions_json)

        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset.json not found: {dataset_path}")
        if not predictions_path.exists():
            raise FileNotFoundError(f"predictions json not found: {predictions_path}")

        with dataset_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)
        with predictions_path.open("r", encoding="utf-8") as f:
            predictions = json.load(f)

        dataset_samples = {s["sample_id"]: s for s in dataset.get("samples", [])}
        prediction_samples = {
            s["sample_id"]: s for s in predictions.get("samples", [])
        }

        total_gt = 0
        total_pred = 0
        total_correct = 0
        per_sample_metrics = []

        for sample_id, sample in dataset_samples.items():
            gt_counter = Counter()
            for pickable in sample.get("pickables", []):
                key = (pickable.get("color_name"), pickable.get("shape"))
                gt_counter[key] += 1

            pred_counter = Counter()
            detection_sample = prediction_samples.get(sample_id, {})
            for detection in detection_sample.get("detections", []):
                key = (detection.get("color"), detection.get("shape"))
                pred_counter[key] += 1

            sample_correct = sum(
                min(gt_counter[key], pred_counter.get(key, 0)) for key in gt_counter
            )
            sample_gt_total = sum(gt_counter.values())
            sample_pred_total = sum(pred_counter.values())

            total_gt += sample_gt_total
            total_pred += sample_pred_total
            total_correct += sample_correct

            per_sample_metrics.append(
                {
                    "sample_id": sample_id,
                    "ground_truth": sample_gt_total,
                    "predicted": sample_pred_total,
                    "correct": sample_correct,
                    "accuracy": sample_correct / sample_gt_total
                    if sample_gt_total
                    else 0.0,
                }
            )

        overall_accuracy = total_correct / total_gt if total_gt else 0.0
        metrics = {
            "dataset": str(dataset_path),
            "predictions": str(predictions_path),
            "total_ground_truth": total_gt,
            "total_predictions": total_pred,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "per_sample": per_sample_metrics,
        }

        print("=== Accuracy Summary ===")
        print(f"Ground truth objects: {total_gt}")
        print(f"Predicted objects:    {total_pred}")
        print(f"Correct matches:      {total_correct}")
        print(f"Accuracy:             {overall_accuracy:.2%}")

        return metrics
