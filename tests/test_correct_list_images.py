import numpy as np

from timepix_geometry_correction.correct import TimepixGeometryCorrection


def test_correct_with_list_images():
    # Create two dummy images of shape (10, 10)
    img1 = np.ones((10, 10), dtype=np.uint8)
    img2 = np.zeros((10, 10), dtype=np.uint8)
    # Instantiate the corrector with a list of images
    corrector = TimepixGeometryCorrection(raw_images=[img1, img2])
    result = corrector.correct()
    # Check that the result is a numpy array of shape (2, new_height, new_width)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2
    # Check that the first image in the result is not all zeros (since img1 is ones)
    assert np.any(result[0])
    # Check that the second image in the result is an array (should be processed)
    assert isinstance(result[1], np.ndarray)
    # Optionally, check that the dtype matches
    assert result.dtype == img1.dtype
