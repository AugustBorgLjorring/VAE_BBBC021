import numpy as np


def crop_image(image):
    """
    Crops an image from (68, 68, 3) to (64, 68, 3) by removing
    two pixels from the top and bottom.

    Returns the cropped image if shape matches, otherwise None.
    """
    if image.shape == (68, 68, 3):
        return image[2:-2, :, :]
    else:
        print(f"Skipping image with unexpected shape: {image.shape}")
        return None
    

    