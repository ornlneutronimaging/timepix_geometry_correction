import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import shift

from timepix_geometry_correction.config import default_config_timepix1
from timepix_geometry_correction.loading import load_tiff_image

chip_size = (256, 256)


class TimepixGeometryCorrection:
    """Apply geometry corrections to Timepix quad-chip detector images.

    A Timepix quad-chip detector is composed of four 256 × 256 pixel chips
    arranged in a 2 × 2 grid::

        +--------+--------+
        | chip 2 | chip 1 |
        | (ref)  | (top R)|
        +--------+--------+
        | chip 3 | chip 4 |
        | (bot L)| (bot R)|
        +--------+--------+

    Manufacturing tolerances cause slight misalignments between chips.
    This class corrects those misalignments by:

    1. **Shift correction** – translating each chip by its measured
       (xoffset, yoffset) using cubic spline interpolation.
    2. **Gap interpolation** – filling the resulting inter-chip gaps
       with linearly (and bilinearly at the centre) interpolated values.

    Parameters
    ----------
    raw_images : numpy.ndarray or list of numpy.ndarray, optional
        One or more 512 × 512 raw detector images provided as arrays.
    images_path : str or list of str, optional
        File path(s) to TIFF images on disk.  Mutually exclusive with
        *raw_images*.
    config : dict, optional
        Per-chip offset configuration.  Each key (``"chip1"`` … ``"chip4"``)
        maps to a dict with ``"xoffset"`` and ``"yoffset"`` (in pixels).
        Defaults to :data:`default_config_timepix1` when *None*.

    Raises
    ------
    ValueError
        If neither *raw_images* nor *images_path* is supplied.

    Examples
    --------
    >>> corrector = TimepixGeometryCorrection(
    ...     images_path="data/siemens_star.tif"
    ... )
    >>> corrected = corrector.correct()
    """

    list_images = None
    list_images_path = None

    chip_size = None

    def __init__(self, raw_images=None, images_path=None, config=None):
        if raw_images is not None:
            if isinstance(raw_images, list):
                self.list_images = raw_images
            else:
                self.list_images = [raw_images]

        elif images_path is not None:
            if isinstance(images_path, list):
                self.list_images_path = images_path
            else:
                self.list_images_path = [images_path]
        else:
            raise ValueError("Either raw_images or list_images_path must be provided.")

        if config is None:
            self.config = default_config_timepix1
        else:
            self.config = config

    def correct(self, display=False):
        """Run the full correction pipeline on every loaded image.

        For each image the method applies, in order:

        1. Sub-pixel shift correction (see :meth:`apply_shift_correction`).
        2. Inter-chip gap interpolation (see :meth:`apply_interpolation_correction`).

        Parameters
        ----------
        display : bool, optional
            If *True*, show a side-by-side matplotlib plot of the original and
            corrected image for every file loaded from disk.  Has no effect
            when images are supplied as arrays.  Default is *False*.

        Returns
        -------
        corrected_list_images : numpy.ndarray
            3-D array of shape ``(N, H, W)`` where *N* is the number of
            images and *H × W* is the corrected image size.

        Raises
        ------
        ValueError
            If neither raw images nor image paths were provided at init.
        """

        if self.list_images is not None:
            self.chip_size = (self.list_images[0].shape[0] // 2, self.list_images[0].shape[1] // 2)
            first_image = self.apply_correction(self.list_images[0])

            # create an empty numpy array of n images with same size as first_image
            corrected_list_images = np.empty((len(self.list_images), *first_image.shape), dtype=first_image.dtype)
            corrected_list_images[0] = first_image
            for index, _data in enumerate(self.list_images[1:]):
                corrected_image = self.apply_correction(_data)
                corrected_list_images[index + 1] = corrected_image
            return corrected_list_images

        elif self.list_images_path is not None:
            # Load the raw image from the specified path and perform correction
            raw_image = load_tiff_image(self.list_images_path[0])
            self.chip_size = (raw_image.shape[0] // 2, raw_image.shape[1] // 2)

            corrected_image = self.apply_correction(raw_image)
            corrected_list_images = np.empty(
                (len(self.list_images_path), *corrected_image.shape), dtype=corrected_image.dtype
            )
            corrected_list_images[0] = corrected_image
            for index, _data in enumerate(self.list_images_path[1:]):
                raw_image = load_tiff_image(_data)
                corrected_image = self.apply_correction(raw_image)
                corrected_list_images[index + 1] = corrected_image

                if display:
                    fig, axs = plt.subplots(ncols=2, figsize=(15, 10))
                    axs[0].imshow(raw_image, vmin=0.55, vmax=0.95, cmap="viridis")
                    axs[0].set_title("Original Image")
                    axs[1].imshow(corrected_image, vmin=0.55, vmax=0.95, cmap="viridis")
                    axs[1].set_title("Corrected Image")
                    plt.show()

            return corrected_list_images

        else:
            raise ValueError("No raw image or path provided for correction.")

    def apply_correction(self, image):
        """Apply the complete geometry correction to a single image.

        Sequentially runs shift correction followed by gap interpolation.

        Parameters
        ----------
        image : numpy.ndarray
            A 2-D raw detector image (typically 512 × 512).

        Returns
        -------
        numpy.ndarray
            The corrected image with inter-chip gaps filled.
        """
        original_dtype = image.dtype
        image = self.apply_shift_correction(image, self.config)
        image = self.apply_interpolation_correction(image, self.config)
        return image.astype(original_dtype, copy=False)

    def apply_shift_correction(self, image, shift_config):
        """Translate each chip by its configured (x, y) offset.

        Chip 2 (top-left) is used as the fixed reference.  The remaining
        three chips are shifted using :func:`scipy.ndimage.shift` with
        third-order (cubic) spline interpolation so that sub-pixel offsets
        are handled smoothly.

        Parameters
        ----------
        image : numpy.ndarray
            A 2-D raw detector image.  NaN and Inf values are replaced
            with 0 before processing.
        shift_config : dict
            Per-chip offset dictionary.  Each entry must contain
            ``"xoffset"`` (column shift) and ``"yoffset"`` (row shift)
            in pixel units.

        Returns
        -------
        new_image : numpy.ndarray
            Image with each chip shifted to its corrected position.
            Gaps created by the shifts are zero-filled at this stage.
        """
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0

        # create an empty array for new image
        # new_image = np.zeros_like(image)
        new_image = np.zeros((image.shape[0], image.shape[1]))

        # chip 2 (fixed one)
        new_image[0:256, 0:256] = image[0:256, 0:256]

        # chip 1
        region = image[0:256, 256:]
        chips1_shift = (shift_config["chip1"]["yoffset"], shift_config["chip1"]["xoffset"])
        shifted_data = shift(region, shift=chips1_shift, order=3)
        new_image[0:256, 256:] = shifted_data

        # chip 3
        region = image[256:, 0:256]
        chips3_shift = (shift_config["chip3"]["yoffset"], shift_config["chip3"]["xoffset"])
        shifted_data = shift(region, shift=chips3_shift, order=3)
        new_image[256:, 0:256] = shifted_data

        # chip 4
        region = image[256:, 256:]
        chips4_shift = (shift_config["chip4"]["yoffset"], shift_config["chip4"]["xoffset"])
        shifted_data = shift(region, shift=chips4_shift, order=3)
        new_image[256:, 256:] = shifted_data

        return new_image

    def interpolate_vertical_chip2_chip1_gap(self, filled, h, w, x_gap_top):
        """Fill the vertical gap in the top row (between chip 2 and chip 1).

        Linearly interpolates along each row between the last column of
        chip 2 and the first column of chip 1 to fill
        ``filled[0:h, w:w+x_gap_top]``.

        Parameters
        ----------
        filled : numpy.ndarray
            Mutable 2-D image array (modified in-place).
        h : int
            Chip height in pixels (typically 256).
        w : int
            Chip width in pixels (typically 256).
        x_gap_top : int
            Width of the gap (number of pixel columns) between chip 2
            and chip 1.
        """
        if x_gap_top > 0:
            left = filled[0:h, w - 1]
            right = filled[0:h, w + x_gap_top]
            for i in range(x_gap_top):
                t = (i + 1) / (x_gap_top + 1)
                filled[0:h, w + i] = (1 - t) * left + t * right

    def interpolate_horizontal_chip2_chip3_gap(self, filled, h, w, y_gap_left):
        """Fill the horizontal gap in the left column (between chip 2 and chip 3).

        Linearly interpolates along each column between the last row of
        chip 2 and the first row of chip 3 to fill
        ``filled[h:h+y_gap_left, 0:w]``.

        Parameters
        ----------
        filled : numpy.ndarray
            Mutable 2-D image array (modified in-place).
        h : int
            Chip height in pixels.
        w : int
            Chip width in pixels.
        y_gap_left : int
            Height of the gap (number of pixel rows) between chip 2
            and chip 3.
        """
        if y_gap_left > 0:
            above = filled[h - 1, 0:w]
            below = filled[h + y_gap_left, 0:w]
            for j in range(y_gap_left):
                t = (j + 1) / (y_gap_left + 1)
                filled[h + j, 0:w] = (1 - t) * above + t * below

    def interpolate_horizontal_chip1_chip4_gap(self, filled, h, w, y_gap_right, max_x_gap):
        """Fill the horizontal gap in the right column (between chip 1 and chip 4).

        Linearly interpolates along each column between the last row of
        chip 1 and the first row of chip 4, covering
        ``filled[h:h+y_gap_right, w+max_x_gap:2*w]``.
        The corner region (``cols w:w+max_x_gap``) is excluded here and
        handled by :meth:`interpolate_corner_intersection`.

        Parameters
        ----------
        filled : numpy.ndarray
            Mutable 2-D image array (modified in-place).
        h : int
            Chip height in pixels.
        w : int
            Chip width in pixels.
        y_gap_right : int
            Height of the gap (number of pixel rows) between chip 1
            and chip 4.
        max_x_gap : int
            Maximum horizontal gap across both top and bottom chip pairs.
            Used to avoid overwriting the corner intersection region.
        """
        if y_gap_right > 0:
            above = filled[h - 1, w + max_x_gap : 2 * w]
            below = filled[h + y_gap_right, w + max_x_gap : 2 * w]
            for j in range(y_gap_right):
                t = (j + 1) / (y_gap_right + 1)
                filled[h + j, w + max_x_gap : 2 * w] = (1 - t) * above + t * below

    def interpolate_vertical_chip3_chip4_gap(self, filled, h, w, x_gap_bot, max_y_gap):
        """Fill the vertical gap in the bottom row (between chip 3 and chip 4).

        Linearly interpolates along each row between the last column of
        chip 3 and the first column of chip 4, covering
        ``filled[h+max_y_gap:2*h, w:w+x_gap_bot]``.
        The corner region (``rows h:h+max_y_gap``) is excluded here and
        handled by :meth:`interpolate_corner_intersection`.

        Parameters
        ----------
        filled : numpy.ndarray
            Mutable 2-D image array (modified in-place).
        h : int
            Chip height in pixels.
        w : int
            Chip width in pixels.
        x_gap_bot : int
            Width of the gap (number of pixel columns) between chip 3
            and chip 4.
        max_y_gap : int
            Maximum vertical gap across both left and right chip pairs.
            Used to avoid overwriting the corner intersection region.
        """
        if x_gap_bot > 0:
            left = filled[h + max_y_gap : 2 * h, w - 1]
            right = filled[h + max_y_gap : 2 * h, w + x_gap_bot]
            for i in range(x_gap_bot):
                t = (i + 1) / (x_gap_bot + 1)
                filled[h + max_y_gap : 2 * h, w + i] = (1 - t) * left + t * right

    def interpolate_corner_intersection(self, filled, h, w, max_x_gap, max_y_gap):
        """Fill the central corner where all four chips meet.

        Uses bilinear interpolation from the four nearest corner pixels
        (one from each chip) to smoothly fill the rectangular region
        ``filled[h:h+max_y_gap, w:w+max_x_gap]``.

        Parameters
        ----------
        filled : numpy.ndarray
            Mutable 2-D image array (modified in-place).
        h : int
            Chip height in pixels.
        w : int
            Chip width in pixels.
        max_x_gap : int
            Width of the corner region (max horizontal gap).
        max_y_gap : int
            Height of the corner region (max vertical gap).

        Notes
        -----
        The four anchor pixels used for bilinear weighting are:

        - **top-left**  : ``filled[h-1, w-1]``        (chip 2 corner)
        - **top-right** : ``filled[h-1, w+max_x_gap]`` (chip 1 edge)
        - **bottom-left** : ``filled[h+max_y_gap, w-1]`` (chip 3 edge)
        - **bottom-right**: ``filled[h+max_y_gap, w+max_x_gap]`` (chip 4 corner)
        """
        if max_x_gap > 0 and max_y_gap > 0:
            tl = filled[h - 1, w - 1]  # chip 2 corner
            tr = filled[h - 1, w + max_x_gap]  # chip 1 side
            bl = filled[h + max_y_gap, w - 1]  # chip 3 side
            br = filled[h + max_y_gap, w + max_x_gap]  # chip 4 corner

            for j in range(max_y_gap):
                ty = (j + 1) / (max_y_gap + 1)
                for i in range(max_x_gap):
                    tx = (i + 1) / (max_x_gap + 1)
                    filled[h + j, w + i] = (
                        tl * (1 - tx) * (1 - ty) + tr * tx * (1 - ty) + bl * (1 - tx) * ty + br * tx * ty
                    )

    def apply_interpolation_correction(self, image, shift_config):
        """Fill gaps between shifted chips using linear interpolation.

        After shift correction, zero-filled gaps appear at chip boundaries
        wherever a chip was translated away from its neighbour.  This method
        fills those gaps in five steps:

        1. Vertical gap in the top row (chip 2 | chip 1) — linear.
        2. Horizontal gap in the left column (chip 2 / chip 3) — linear.
        3. Horizontal gap in the right column (chip 1 / chip 4) — linear.
        4. Vertical gap in the bottom row (chip 3 | chip 4) — linear.
        5. Central corner intersection — bilinear.

        Parameters
        ----------
        image : numpy.ndarray
            2-D shift-corrected image containing zero-filled gaps.
        shift_config : dict
            Per-chip offset dictionary (same format as the class *config*).

        Returns
        -------
        filled : numpy.ndarray
            Copy of *image* with all inter-chip gaps filled by
            interpolation.

        See Also
        --------
        interpolate_vertical_chip2_chip1_gap : Step 1.
        interpolate_horizontal_chip2_chip3_gap : Step 2.
        interpolate_horizontal_chip1_chip4_gap : Step 3.
        interpolate_vertical_chip3_chip4_gap : Step 4.
        interpolate_corner_intersection : Step 5.
        """
        filled = image.copy().astype(float)
        if self.chip_size is not None:
            h, w = self.chip_size
        else:
            h, w = chip_size

        # Offsets that define the gap sizes at each boundary. These offsets may be
        # fractional (sub-pixel), but gap sizes used for indexing/interpolation
        # must be integers, so we derive integer gap widths/heights here.
        x_offset_top = shift_config["chip1"]["xoffset"]  # vertical gap width  (top half)
        y_offset_left = shift_config["chip3"]["yoffset"]  # horizontal gap height (left half)
        x_offset_bot = shift_config["chip4"]["xoffset"]  # vertical gap width  (bottom half)
        y_offset_right = shift_config["chip4"]["yoffset"]  # horizontal gap height (right half)

        x_gap_top = int(round(x_offset_top))
        y_gap_left = int(round(y_offset_left))
        x_gap_bot = int(round(x_offset_bot))
        y_gap_right = int(round(y_offset_right))
        max_x_gap = max(x_gap_top, x_gap_bot)
        max_y_gap = max(y_gap_left, y_gap_right)

        self.interpolate_vertical_chip2_chip1_gap(filled, h, w, x_gap_top)
        self.interpolate_horizontal_chip2_chip3_gap(filled, h, w, y_gap_left)
        self.interpolate_horizontal_chip1_chip4_gap(filled, h, w, y_gap_right, max_x_gap)
        self.interpolate_vertical_chip3_chip4_gap(filled, h, w, x_gap_bot, max_y_gap)
        self.interpolate_corner_intersection(filled, h, w, max_x_gap, max_y_gap)

        return filled


if __name__ == "__main__":
    o_corrector = TimepixGeometryCorrection(
        images_path=["notebooks/data/siemens_star.tif", "notebooks/data/rectangle_grid.tif"]
    )
    corrected = o_corrector.correct(display=True)
