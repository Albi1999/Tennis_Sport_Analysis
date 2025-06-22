import cv2
import numpy as np

def test_img_values(img):
    """
    Test the pixel values of the image.
    The function checks if the pixel values are in the range [0, 255].
    """
    
    assert np.max(img) <= 255, "Image pixel values should be in the range [0, 255]"
    assert np.min(img) >= 0, "Image pixel values should be in the range [0, 255]"
    
    return True

def test_heatmap_values(heatmap):
    """
    Test the heatmap values.
    The function checks if the heatmap values are in the range [0, 1].
    """
    
    assert np.max(heatmap) <= 1, "Heatmap values should be in the range [0, 1]"
    assert np.min(heatmap) >= 0, "Heatmap values should be in the range [0, 1]"
    
    return True



def convert_to_heatmap_values(img):
    """
    Convert the image to a heatmap by normalizing the pixel values.
    The function normalizes the pixel values to the range [0, 1].
    The input image is expected to be in the range [0, 255].
    """
    
    test_img_values(img)  # Test the image values
    
    # Normalize the image to the range [0, 1]
    heatmap = img.astype(np.float32) / 255.0
    
    return heatmap


def convert_to_pixel_values(heatmap):
    """
    Convert the heatmap back to pixel values.
    The function converts the heatmap values back to the original pixel value range [0, 255].
    The input heatmap is expected to be in the range [0, 1].
    """
    
    test_heatmap_values(heatmap)  # Test the heatmap values
    
    # Convert the heatmap values back to pixel values
    img = (heatmap * 255).astype(np.uint8)
    
    return img
    
def apply_colormap(img, colormap=cv2.COLORMAP_HOT):
    """
    Apply a colormap to the image.
    The function applies the specified colormap to the image.
    The input image is expected to be in the range [0, 255].
    """
    
    test_img_values(img)  # Test the image values

    
    # Apply the colormap
    colored_img = cv2.applyColorMap(img, colormap)
    
    
    return cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB) # For warm colors

def compute_score_heatmap(player_heatmap, ball_heatmap, alpha=0.15, beta=0, gamma=1):
    """
    Compute the score heatmap based on player and ball heatmaps.
    The function combines the player and ball heatmaps using the specified weights.
    The heatmaps are expected to be in the range [0, 1].
    The alpha, beta, and gamma parameters control the contribution of each heatmap to the final score heatmap.
    - alpha: Weight for the player heatmap (default: 0.15)
    - beta: Weight for the ball heatmap (default: 0) 
    - gamma: Weight for the interaction term (default: 1)
    """

    test_heatmap_values(player_heatmap)     # Test the player heatmap values
    test_heatmap_values(ball_heatmap)       # Test the ball heatmap values


    # Combine the player and ball heatmaps using the specified weights
    score_heatmap = alpha * player_heatmap + beta * ball_heatmap + gamma * (player_heatmap * ball_heatmap)
    score_heatmap = np.clip(score_heatmap, 0, 1)  # Ensure values are in the range [0, 1]
    
    return score_heatmap

def compute_score_probability(score_heatmap):
    """
    Compute the score probability based on the score heatmap.
    The function computes the score probability by finding the maximum value in the score heatmap.
    The score heatmap is expected to be in the range [0, 1].
    """
    
    test_heatmap_values(score_heatmap)  # Test the heatmap values
    
    # Compute the score probability
    score_probability = np.max(score_heatmap) 
    
    return score_probability * 100  # Convert to percentage