import os
import cv2
import torch
import numpy as np

def ncc_horizontal(left_img, right_img, template_center, template_radius, max_disp):
    """
    Compute normalized cross-correlation (NCC) along a single row,
    scanning only horizontally within [-max_disp, max_disp].

    Args:
        left_img: H x W grayscale left image
        right_img: H x W grayscale right image
        template_center: (x, y) tuple in right image
        template_radius: half-size of square template (template size = 2*radius+1)
        max_disp: maximum horizontal distance to search (disparity)

    Returns:
        best_x: x-coordinate of best match in left image
        best_disp: disparity = x_left - x_right
        max_ncc: maximum NCC score
    """
    H, W = left_img.shape
    x, y = template_center
    th = tw = template_radius * 2 + 1

    # --- extract template from right image ---
    template = right_img[y-template_radius:y+template_radius+1,
                        x-template_radius:x+template_radius+1].astype(np.float32)

    # mean and std of template
    t_mean = np.mean(template)
    t_std = np.std(template)
    if t_std < 1e-6:
        t_std = 1e-6  # prevent division by zero

    best_ncc = -1
    best_x = x

    # define horizontal search range
    x_start = max(0, x - max_disp)
    x_end   = min(W - tw, x + max_disp)

    # only scan this row
    row = y
    for xr in range(x_start, x_end + 1):
        patch = left_img[row-template_radius:row+template_radius+1,
                          xr:xr+tw].astype(np.float32)
        p_mean = np.mean(patch)
        p_std  = np.std(patch)
        if p_std < 1e-6:
            p_std = 1e-6

        # NCC formula
        ncc = np.sum((template - t_mean) * (patch - p_mean)) / (t_std * p_std * tw * th)

        if ncc > best_ncc:
            best_ncc = ncc
            best_x = xr

    best_disp = best_x - x  # disparity = x_left - x_right
    return best_x, best_disp, best_ncc


# --------------------- Example usage ---------------------

# load grayscale images
left_tensor = torch.load(os.path.join("test_images", "left_camera_rgb.pt"), weights_only=True)
right_tensor = torch.load(os.path.join("test_images", "right_camera_rgb.pt"), weights_only=True)
left = left_tensor.cpu().numpy()
right = right_tensor.cpu().numpy()
if left.ndim == 3 and left.shape[0] == 3:
    left = left.transpose(1, 2, 0)
    right = right.transpose(1, 2, 0)
if left.ndim == 3:
    left = cv2.cvtColor(left.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right.astype(np.uint8), cv2.COLOR_RGB2GRAY)

template_radius = 22
max_disp = 200

x = 1500
y = 820

best_x, disparity, score = ncc_horizontal(left, right, (x, y), template_radius, max_disp)
print(f"Selected point: ({x},{y})")
print(f"Best match: ({best_x},{y}), disparity: {disparity}, NCC score: {score:.4f}")

# --- visualization ---
left_debug = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
right_debug = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

H, W = left.shape
th = tw = template_radius * 2 + 1

# Calculate scan range (same as in ncc_horizontal)
x_start = max(0, x - max_disp)
x_end = min(W - tw, x + max_disp)

# Draw scan range on LEFT image (cyan horizontal line with markers)
scan_y = y  # scanning along this row
cv2.line(left_debug, (x_start, scan_y), (x_end + tw, scan_y), (255, 255, 0), 2)  # cyan line
cv2.circle(left_debug, (x_start, scan_y), 6, (255, 0, 255), -1)  # magenta dot = start
cv2.circle(left_debug, (x_end + tw, scan_y), 6, (0, 165, 255), -1)  # orange dot = end

# Draw template region on RIGHT image (red rectangle)
cv2.rectangle(right_debug, (x-template_radius, y-template_radius), 
              (x+template_radius+1, y+template_radius+1), (0,0,255), 2)

# Draw matched region on LEFT image (green rectangle)
cv2.rectangle(left_debug, (best_x, y-template_radius), 
              (best_x+tw, y+template_radius+1), (0,255,0), 2)

# Stack images side by side
debug = np.hstack([right_debug, left_debug])

# Resize to 1920x1080
debug = cv2.resize(debug, (1920, 1080), interpolation=cv2.INTER_LINEAR)

# Add labels
cv2.putText(debug, "RIGHT (template source)", (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.putText(debug, "LEFT (match found)", (980, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.putText(debug, f"Disparity: {disparity}px | NCC: {score:.4f}", (20, 1050), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
cv2.putText(debug, "Scan: magenta=start, orange=end", (1400, 1050), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

cv2.imshow("Stereo Match Result", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()