import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.append('../')
from utils import convert_to_heatmap_values, convert_to_pixel_values, apply_colormap, compute_score_heatmap, compute_score_probability, test_img_values, test_heatmap_values


# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Ball position (x, y) in the image
ball_pos = (50, 50)             # Ball position (use minicourt Ball detections)
bl_radius = 50                  # Radius of the ball landing area

# Ball Heatmap
ball_img = deepcopy(img)

# Approach using NumPy vectorization 
y, x = np.ogrid[:img.shape[0], :img.shape[1]]
ball_distance = np.sqrt((x - ball_pos[0])**2 + (y - ball_pos[1])**2)

# Normalize distance to range 0-255
max_distance = np.sqrt(ball_img.shape[0]**2 + ball_img.shape[1]**2)
ball_intensity = 255 * ball_distance / max_distance

# Create a mask for the ball's distance using ball landing area
distance_mask = ball_distance >= bl_radius

# Assign the same intensity to all three channels
ball_img[:, :, 0] = ball_intensity
ball_img[:, :, 1] = ball_intensity
ball_img[:, :, 2] = ball_intensity

# Invert the intensity
ball_img = 255 - ball_img

# Apply the mask to the ball image
ball_img[distance_mask] = 0

# Scale the ball heatmap to the range 0-255
ball_heatmap = convert_to_heatmap_values(ball_img)

# Colorize the heatmaps
colored_ball_heatmap = apply_colormap(ball_img)