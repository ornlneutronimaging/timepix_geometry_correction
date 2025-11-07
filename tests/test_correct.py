from pathlib import Path

import numpy as np
import pytest

from timepix_geometry_correction import chip_size
from timepix_geometry_correction.config import default_config_timepix1 as config
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
        """Fixture providing a synthetic test image with known properties."""
        # Create a 512x512 image (2x2 chips of 256x256 each)
        image = np.zeros((512, 512), dtype=np.uint16)

        # Add distinct patterns to each chip for easy identification
        # Chip 1 (top right): diagonal pattern
        image[0:256, 256:512] = np.arange(256)[:, np.newaxis] + np.arange(256)[np.newaxis, :]

        # Chip 2 (top left): vertical stripes
        image[0:256, 0:256] = np.tile(np.arange(256), (256, 1))

        # Chip 3 (bottom left): horizontal stripes
        image[256:512, 0:256] = np.tile(np.arange(256)[:, np.newaxis], (1, 256))

        # Chip 4 (bottom right): checkerboard pattern
        x, y = np.meshgrid(np.arange(256), np.arange(256))
        image[256:512, 256:512] = ((x // 32) + (y // 32)) % 2 * 255

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
        expected_height = sample_image.shape[0] + max([config[chip]["yoffset"] for chip in config])
        expected_width = sample_image.shape[1] + max([config[chip]["xoffset"] for chip in config])
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
        expected_height = original.shape[0] + max([config[chip]["yoffset"] for chip in config])
        expected_width = original.shape[1] + max([config[chip]["xoffset"] for chip in config])
        assert corrected.shape == (1, expected_height, expected_width)

        # Check that output is not all zeros
        assert np.any(corrected > 0)

    def test_apply_correction_chip_placement(self, sample_image):
        """Test that chips are placed correctly in the corrected image."""
        corrector = TimepixGeometryCorrection(raw_images=sample_image)
        corrected = corrector.apply_correction(sample_image)

        # Extract original chip data
        original_chip1 = sample_image[0 : chip_size[0], chip_size[1] :]
        original_chip2 = sample_image[0 : chip_size[0], 0 : chip_size[1]]
        original_chip3 = sample_image[chip_size[0] :, 0 : chip_size[1]]
        original_chip4 = sample_image[chip_size[0] :, chip_size[1] :]

        # Check chip2 placement (reference chip at origin)
        corrected_chip2 = corrected[0 : chip_size[0], 0 : chip_size[1]]
        np.testing.assert_array_equal(corrected_chip2, original_chip2)

        # Check chip1 placement (with offsets)
        chip1_start_y = config["chip1"]["yoffset"]
        chip1_end_y = chip_size[0] + config["chip1"]["yoffset"]
        chip1_start_x = chip_size[1] + config["chip1"]["xoffset"]
        chip1_end_x = 2 * chip_size[1] + config["chip1"]["xoffset"]
        corrected_chip1 = corrected[chip1_start_y:chip1_end_y, chip1_start_x:chip1_end_x]
        np.testing.assert_array_equal(corrected_chip1, original_chip1)

        # Check chip3 placement
        chip3_start_y = chip_size[0] + config["chip3"]["yoffset"]
        chip3_end_y = 2 * chip_size[0] + config["chip3"]["yoffset"]
        chip3_start_x = config["chip3"]["xoffset"]
        chip3_end_x = chip_size[1] + config["chip3"]["xoffset"]
        corrected_chip3 = corrected[chip3_start_y:chip3_end_y, chip3_start_x:chip3_end_x]
        np.testing.assert_array_equal(corrected_chip3, original_chip3)

        # Check chip4 placement
        chip4_start_y = chip_size[0] + config["chip4"]["yoffset"]
        chip4_end_y = 2 * chip_size[0] + config["chip4"]["yoffset"]
        chip4_start_x = chip_size[1] + config["chip4"]["xoffset"]
        chip4_end_x = 2 * chip_size[1] + config["chip4"]["xoffset"]
        corrected_chip4 = corrected[chip4_start_y:chip4_end_y, chip4_start_x:chip4_end_x]
        np.testing.assert_array_equal(corrected_chip4, original_chip4)


#     def test_correction_preserves_dtype(self, sample_image):
#         """Test that correction preserves the original image data type."""
#         corrector = TimepixGeometryCorrection(raw_images=sample_image)
#         corrected = corrector.correct()
#         assert corrected.dtype == sample_image.dtype

#     def test_gap_interpolation_between_chips_1_and_2(self):
#         """Test gap interpolation between chips 1 and 2."""
#         # Create a simple test image with known values
#         test_image = np.zeros((512, 512), dtype=np.uint16)
#         # Set specific values at chip boundaries for testing interpolation
#         test_image[100, 255] = 100  # Left boundary (chip 2)
#         test_image[100, 257] = 200  # Right boundary (chip 1, after offset)

#         corrector = TimepixGeometryCorrection(raw_images=test_image)
#         corrected = corrector.correct()

#         # Check that interpolation occurred in the gap
#         gap_start = chip_size[1]
#         gap_end = chip_size[1] + config["chip1"]["xoffset"]
#         gap_values = corrected[100, gap_start:gap_end]

#         # Gap should be filled with interpolated values
#         assert not np.all(gap_values == 0)

#     def test_siemens_star_specific_properties(self, siemens_star_path):
#         """Test correction specifically with the Siemens star image."""
#         corrector = TimepixGeometryCorrection(images_path=siemens_star_path)
#         corrected = corrector.correct()

#         # Load original to compare
#         original = load_tiff_image(siemens_star_path)

#         # Basic sanity checks
#         assert corrected.shape[0] > original.shape[0]  # Height should increase due to offsets
#         assert corrected.shape[1] > original.shape[1]  # Width should increase due to offsets

#         # Check that the correction maintains the overall intensity distribution
#         original_nonzero = original[original > 0]
#         corrected_nonzero = corrected[corrected > 0]

#         if len(original_nonzero) > 0 and len(corrected_nonzero) > 0:
#             # The corrected image should have similar intensity characteristics
#             assert np.min(corrected_nonzero) >= np.min(original_nonzero)
#             assert np.max(corrected_nonzero) <= np.max(original_nonzero)

#     def test_correction_is_deterministic(self, sample_image):
#         """Test that correction produces consistent results."""
#         corrector1 = TimepixGeometryCorrection(raw_images=sample_image)
#         corrector2 = TimepixGeometryCorrection(raw_images=sample_image)

#         corrected1 = corrector1.correct()
#         corrected2 = corrector2.correct()

#         np.testing.assert_array_equal(corrected1, corrected2)

#     def test_individual_gap_correction_methods(self, sample_image):
#         """Test individual gap correction methods."""
#         corrector = TimepixGeometryCorrection(raw_images=sample_image)
#         corrected = corrector.apply_correction(sample_image)

#         # Create a copy to test individual methods
#         test_image = corrected.copy()

#         # Test that the gap correction methods don't crash
#         corrector.correct_between_chips_1_and_2(test_image)
#         corrector.correct_between_chips_2_and_3(test_image)
#         corrector.correct_between_chips_1_and_4(test_image)
#         corrector.correct_between_chips_3_and_4(test_image)

#         # The image should still have the same shape
#         assert test_image.shape == corrected.shape

#     def test_provide_custom_config(self, sample_image):
#         """Test that a custom configuration can be provided."""
#         custom_config = {
#             "chip1": {"xoffset": 5, "yoffset": 10},
#             "chip2": {"xoffset": 5, "yoffset": 10},
#             "chip3": {"xoffset": 5, "yoffset": 10},
#             "chip4": {"xoffset": 5, "yoffset": 10},
#         }
#         corrector = TimepixGeometryCorrection(raw_images=sample_image, config=custom_config)
#         corrected = corrector.correct()

#         # Check that the corrected image is produced with the custom config
#         assert corrected is not None
#         assert corrected.shape == (512 + 10, 512 + 5)
#         assert np.all(corrected[100, 5:10] == [5, 6, 7, 8, 9])  # Check that the gap is filled


# def test_config_integrity():
#     """Test that the configuration values are sensible."""
#     assert isinstance(config, dict)
#     assert len(config) == 4

#     required_chips = ["chip1", "chip2", "chip3", "chip4"]
#     for chip in required_chips:
#         assert chip in config
#         assert "xoffset" in config[chip]
#         assert "yoffset" in config[chip]
#         assert isinstance(config[chip]["xoffset"], int)
#         assert isinstance(config[chip]["yoffset"], int)
#         assert config[chip]["xoffset"] >= 0
#         assert config[chip]["yoffset"] >= 0


# def test_chip_size_integrity():
#     """Test that chip size is properly defined."""
#     assert isinstance(chip_size, tuple)
#     assert len(chip_size) == 2
#     assert all(isinstance(x, int) for x in chip_size)
#     assert all(x > 0 for x in chip_size)


# def test_load_siemens_star_image():
#     """Test that the Siemens star image can be loaded."""
#     test_data_path = Path(__file__).parent.parent / "notebooks" / "data" / "siemens_star.tif"
#     if not test_data_path.exists():
#         pytest.skip(f"Test data file not found: {test_data_path}")

#     image = load_tiff_image(str(test_data_path))
#     assert isinstance(image, np.ndarray)
#     assert image.ndim == 2  # Should be a 2D image
#     assert image.shape == (512, 512)  # Expected size for 2x2 timepix chips
#     assert image.dtype in [np.uint8, np.uint16, np.uint32, np.float32, np.float64]
