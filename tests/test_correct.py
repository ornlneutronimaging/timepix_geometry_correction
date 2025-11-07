from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from timepix_geometry_correction.correct import TimepixGeometryCorrection
from timepix_geometry_correction.loading import load_tiff_image


class TestTimepixGeometryCorrection:
    """Test class for TimepixGeometryCorrection functionality."""

    @pytest.fixture
    def siemens_star_path(self):
        """Fixture providing path to the siemens star test image."""
        test_data_path = Path(__file__).parent.parent / "notebooks" / "data" / "siemens_star.tif"
        if not test_data_path.exists():
            pytest.skip(f"Test data file not found: {test_data_path}")
        return str(test_data_path)

    @pytest.fixture
    def sample_image(self):
        """Fixture providing an image corrected."""
        corrected_image_path = Path(__file__).parent.parent / "notebooks" / "data" / "corrected_siemens_star.tif"
        image = np.array(Image.open(corrected_image_path))
        return image

    def test_init_with_image(self, sample_image):
        """Test initialization with raw image data."""
        corrector = TimepixGeometryCorrection(raw_images=sample_image)
        assert corrector.list_images is not None
        assert corrector.list_images_path is None
        np.testing.assert_array_equal(corrector.list_images, [sample_image])

    def test_init_with_path(self, siemens_star_path):
        """Test initialization with image file path."""
        corrector = TimepixGeometryCorrection(images_path=siemens_star_path)
        assert corrector.list_images is None
        assert corrector.list_images_path == [siemens_star_path]

    def test_init_without_input(self):
        """Test initialization without image or path should be allowed."""
        # assert that this raises an error
        with pytest.raises(ValueError, match="Either raw_images or list_images_path must be provided."):
            _ = TimepixGeometryCorrection()

    def test_correct_with_image(self, sample_image):
        """Test correction using raw image data."""
        corrector = TimepixGeometryCorrection(raw_images=sample_image)
        corrected = corrector.correct()

        # Check output dimensions
        expected_height = sample_image.shape[0] + int(np.ceil(max([config[chip]["yoffset"] for chip in config])))
        expected_width = sample_image.shape[1] + int(np.ceil(max([config[chip]["xoffset"] for chip in config])))
        assert corrected.shape == (1, expected_height, expected_width)

        # Check that output is not all zeros
        assert np.any(corrected > 0)

    def test_correct_with_path(self, siemens_star_path):
        """Test correction using image file path."""
        corrector = TimepixGeometryCorrection(images_path=siemens_star_path)
        corrected = corrector.correct()

        # Load the original image to compare dimensions
        original = load_tiff_image(siemens_star_path)

        # Check output dimensions
        expected_height = original.shape[0] + int(np.ceil(max([config[chip]["yoffset"] for chip in config])))
        expected_width = original.shape[1] + int(np.ceil(max([config[chip]["xoffset"] for chip in config])))
        assert corrected.shape == (1, expected_height, expected_width)

        # Check that output is not all zeros
        assert np.any(corrected > 0)
