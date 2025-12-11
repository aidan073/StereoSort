import cv2
import numpy as np
import torch
from pathlib import Path

class ShapeColorClassifier:
    def __init__(self):
        # --- absolute color masks in HSV ---
        self.lower_red1, self.upper_red1 = np.array([0, 180, 180]), np.array([10, 255, 255])
        self.lower_red2, self.upper_red2 = np.array([170, 180, 180]), np.array([179, 255, 255])
        self.lower_green, self.upper_green = np.array([50, 180, 180]), np.array([70, 255, 255])
        self.lower_blue, self.upper_blue = np.array([110, 180, 180]), np.array([130, 255, 255])

    def classify_shape(self, contour):
        area = cv2.contourArea(contour)
        if area < 200:
            return None
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        vertices = len(approx)
        circularity = (4 * np.pi * area) / (peri * peri + 1e-5)

        if vertices == 4:
            return "cube"
        elif circularity > 0.7:
            return "cylinder"
        elif vertices == 3:
            return "triangular_prism"
        else:
            return "unknown"

    def classify_color_at_pixel(self, hsv_img, cx, cy):
        hue, sat, val = hsv_img[cy, cx]
        if (hue <= 10 or hue >= 170) and sat > 150 and val > 150:
            return "red", (255, 255, 255)
        elif 50 <= hue <= 70 and sat > 150 and val > 150:
            return "green", (255, 0, 255)
        elif 110 <= hue <= 130 and sat > 150 and val > 150:
            return "blue", (0, 255, 255)
        else:
            return "unknown", (200, 200, 200)

    def load_pt_image(self, path):
        tensor = torch.load(path, map_location="cpu")
        if not torch.is_tensor(tensor):
            raise ValueError(f"{Path(path).name} does not contain a tensor.")
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        tensor = tensor.squeeze()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.max() > 1:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte().cpu().numpy()
        rgb = np.transpose(tensor, (1, 2, 0))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def process_image(self, img_path):
        shapes = []
        img = self.load_pt_image(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_red = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, mask_blue))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            shape = self.classify_shape(c)
            if not shape:
                continue
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            centroid = (cx, cy)
            color_name, draw_color = self.classify_color_at_pixel(hsv, cx, cy)
            shapes.append({"shape": shape, "color": color_name, "centroid": centroid})
        return shapes

    def contour_image(self, image_path):
        return self.process_image(image_path)
