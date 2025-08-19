import numpy as np
from dxchange import read_tiff


def load_tiff_image(file_path):
    """
    Load an image from a TIFF file.

    Parameters:
    file_path (str): Path to the TIFF file.

    Returns:
    np.ndarray: Loaded image.
    """
    return np.array(read_tiff(file_path))
