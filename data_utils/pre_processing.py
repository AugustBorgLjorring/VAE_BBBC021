import numpy as np


def crop_image(image: np.ndarray, pix : int = 2) -> np.ndarray:
    """
    Params:
    image: np.ndarray
        An image with shape (68, 68, 3)

    pix: int
        Number of pixels to crop from the top and bottom of the image.

    Crops an image from (68, 68, 3) to (64, 64, 3) by removing
    two pixels from the top and bottom.

    Returns the cropped image if shape matches, otherwise None.
    """
    if image.shape == (68, 68, 3):
        return image[pix:-pix, pix:-pix, :]
    else:
        print(f"Skipping image with unexpected shape: {image.shape}")
        return None
    


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes an image within each channel.
    im = (im - im.min()) / (im.max() - im.min())
    """
    min = image.min(axis=(0, 1))
    max = image.max(axis=(0, 1))

    image_norm = (image - min) / (max - min)

    return image_norm