import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import shift

from timepix_geometry_correction.config import default_config_timepix1
from timepix_geometry_correction.loading import load_tiff_image


class TimepixGeometryCorrection:
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
        # Apply geometric corrections to the event using self.raw_image or self.raw_image_path

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

    def apply_shift_correction(self, image, shift_config):
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

    def apply_correction(self, image):
        if self.chip_size is None:
            self.chip_size = (image.shape[0] // 2, image.shape[1] // 2)

        # data = {
        #     "chip1": image[0 : self.chip_size[0], self.chip_size[1] :],
        #     "chip2": image[0 : self.chip_size[0], 0 : self.chip_size[1]],
        #     "chip3": image[self.chip_size[0] :, 0 : self.chip_size[1]],
        #     "chip4": image[self.chip_size[0] :, self.chip_size[1] :],
        # }

        new_image = self.apply_shift_correction(image, self.config)

        self.correct_between_chips_1_and_2(new_image)
        self.correct_between_chips_2_and_3(new_image)
        self.correct_between_chips_1_and_4(new_image)
        self.correct_between_chips_3_and_4(new_image)
        self.correct_center_area(new_image)

        return new_image.astype(image.dtype)

    def correct_between_chips_1_and_2(self, new_image):
        # between chips 1 and 2, we need to correct the gap
        # gap is config['chip1']['xoffset'] (horizontal) and config['chip1']['yoffset'] (vertical)
        # y will go from 0 to self.chip_size[0] + config['chip1']['yoffset']
        config = self.config
        chip_size = self.chip_size

        # Precompute ceiled offsets to avoid repeated computation in loop
        xoffset_ceiled = int(np.ceil(config["chip1"]["xoffset"]))
        yoffset_ceiled = int(np.ceil(config["chip1"]["yoffset"]))

        for _y in range(0, chip_size[0] + yoffset_ceiled):
            left_value = new_image[_y, chip_size[0] - 1]
            right_value = new_image[_y, chip_size[0] + xoffset_ceiled]

            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(xoffset_ceiled)
            if left_value == 0:
                list_new_value = np.ones(xoffset_ceiled) * right_value
            elif right_value == 0:
                list_new_value = np.ones(xoffset_ceiled) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, xoffset_ceiled + 1),
                    [0, xoffset_ceiled + 1],
                    [left_value, right_value],
                )

            new_image[_y, chip_size[1] : chip_size[1] + xoffset_ceiled] = list_new_value

    def correct_between_chips_2_and_3(self, new_image):
        # between chips 2 and 3
        # gap is config['chip3']['xoffset'] (horizontal) and config['chip3']['yoffset'] (vertical)
        # x will go from 0 to chip_size[1] + config['chip3']['xoffset']
        config = self.config
        chip_size = self.chip_size

        # Precompute ceiled offsets to avoid repeated computation in loop
        xoffset_ceiled = int(np.ceil(config["chip3"]["xoffset"]))
        yoffset_ceiled = int(np.ceil(config["chip3"]["yoffset"]))

        for _x in range(0, chip_size[1] + xoffset_ceiled):
            left_value = new_image[chip_size[0] - 1, _x]
            right_value = new_image[chip_size[0] + yoffset_ceiled, _x]
            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(yoffset_ceiled)
            if left_value == 0:
                list_new_value = np.ones(yoffset_ceiled) * right_value
            elif right_value == 0:
                list_new_value = np.ones(yoffset_ceiled) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, yoffset_ceiled + 1),
                    [0, yoffset_ceiled + 1],
                    [left_value, right_value],
                )

            new_image[chip_size[0] : chip_size[0] + yoffset_ceiled, _x] = list_new_value

    def correct_between_chips_1_and_4(self, new_image):
        # between chips 1 and 4
        # gap is config['chip4']['xoffset'] - config['chip1']['xoffset']
        # (horizontal) and config['chip4']['yoffset'] - config['chip1']['yoffset'] (vertical)
        # x will go from chip_size[1]+config['chip1']['xoffset'] to 2*chip_size[1]+config['chip1']['xoffset']
        config = self.config
        chip_size = self.chip_size

        # Precompute ceiled offsets to avoid repeated computation in loop
        chip1_xoffset_ceiled = int(np.ceil(config["chip1"]["xoffset"]))
        chip1_yoffset_ceiled = int(np.ceil(config["chip1"]["yoffset"]))
        chip4_yoffset_ceiled = int(np.ceil(config["chip4"]["yoffset"]))

        for _x in range(
            chip_size[1] + chip1_xoffset_ceiled,
            2 * chip_size[1] + chip1_xoffset_ceiled - 3,
        ):
            left_value = new_image[chip_size[0] - 2 + config["chip1"]["yoffset"], _x]
            right_value = new_image[chip_size[0] + chip4_yoffset_ceiled, _x]
            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(chip4_yoffset_ceiled)
            if left_value == 0:
                list_new_value = np.ones(chip4_yoffset_ceiled) * right_value
            elif right_value == 0:
                list_new_value = np.ones(chip4_yoffset_ceiled) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, chip4_yoffset_ceiled + 1),
                    [0, chip4_yoffset_ceiled + chip1_yoffset_ceiled + 1],
                    [left_value, right_value],
                )

            new_image[
                chip_size[0] + chip1_yoffset_ceiled - 1 : chip_size[0]
                + chip4_yoffset_ceiled
                + chip1_yoffset_ceiled
                - 1,
                _x,
            ] = list_new_value

    def correct_between_chips_3_and_4(self, new_image):
        config = self.config
        chip_size = self.chip_size

        # Precompute ceiled offsets to avoid repeated computation in loop
        chip3_yoffset_ceiled = int(np.ceil(config["chip3"]["yoffset"]))
        chip4_xoffset_ceiled = int(np.ceil(config["chip4"]["xoffset"]))

        for _y in range(
            int(chip_size[0] + chip3_yoffset_ceiled + config["chip1"]["yoffset"]),
            int(2 * chip_size[0] + chip3_yoffset_ceiled + config["chip1"]["yoffset"] - 2),
        ):
            left_value = new_image[_y, chip_size[1] - 1]
            right_value = new_image[_y, chip_size[1] + chip4_xoffset_ceiled]
            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(chip4_xoffset_ceiled)
            if left_value == 0:
                list_new_value = np.ones(chip4_xoffset_ceiled) * right_value
            elif right_value == 0:
                list_new_value = np.ones(chip4_xoffset_ceiled) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, chip4_xoffset_ceiled + 1),
                    [0, chip4_xoffset_ceiled + 1],
                    [left_value, right_value],
                )

            new_image[_y, chip_size[1] : chip_size[1] + chip4_xoffset_ceiled] = list_new_value

    def correct_center_area(self, new_image):
        chip_size = self.chip_size
        for _x in range(chip_size[1], chip_size[1] + 3):
            left_value = new_image[chip_size[0] - 1, _x]
            right_value = new_image[chip_size[0] + 2, _x]

            list_new_value = np.interp(np.arange(1, 4), [0, 4], [left_value, right_value])

            new_image[chip_size[0] : chip_size[0] + 3, _x] = list_new_value


if __name__ == "__main__":
    o_corrector = TimepixGeometryCorrection(
        images_path=["notebooks/data/siemens_star.tif", "notebooks/data/rectangle_grid.tif"]
    )
    corrected = o_corrector.correct(display=True)
