import numpy as np
from PIL import Image


def load_tiff_image(file_path):
    """
    Load an image from a TIFF file.

    Parameters:
    file_path (str): Path to the TIFF file.

    Returns:
    np.ndarray: Loaded image.
    """
    with Image.open(file_path) as img:
        return np.array(img)
