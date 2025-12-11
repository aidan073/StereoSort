"""
This file holds functions that allow going from stereo images to ordered target
positions, based on some sorting criteria.
"""

import torch
import cv2 as cv

from typing import List, Tuple


class StereoSort:
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
