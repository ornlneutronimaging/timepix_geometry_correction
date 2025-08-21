import numpy as np

from timepix_geometry_correction import chip_size
from timepix_geometry_correction.config import default_config_timepix1
from timepix_geometry_correction.loading import load_tiff_image


class TimepixGeometryCorrection:
    def __init__(self, raw_image=None, raw_image_path=None, config=None):
        self.raw_image = raw_image
        self.raw_image_path = raw_image_path
        if config is None:
            self.config = default_config_timepix1
        else:
            self.config = config

    def correct(self):
        # Apply geometric corrections to the event using self.raw_image or self.raw_image_path
        if self.raw_image is not None:
            # Perform correction using the raw image directly
            corrected_image = self.apply_correction(self.raw_image)
        elif self.raw_image_path is not None:
            # Load the raw image from the specified path and perform correction
            raw_image = load_tiff_image(self.raw_image_path)
            corrected_image = self.apply_correction(raw_image)
        else:
            raise ValueError("No raw image or path provided for correction.")

        return corrected_image

    def apply_correction(self, image):
        # Placeholder for the actual correction logic
        # For now, just return the image as is

        data = {
            "chip1": image[0 : chip_size[0], chip_size[1] :],
            "chip2": image[0 : chip_size[0], 0 : chip_size[1]],
            "chip3": image[chip_size[0] :, 0 : chip_size[1]],
            "chip4": image[chip_size[0] :, chip_size[1] :],
        }

        # apply correction to each chip
        old_image_height, old_image_width = image.shape

        new_image_height = old_image_height + max([self.config[chip]["yoffset"] for chip in self.config])
        new_image_width = old_image_width + max([self.config[chip]["xoffset"] for chip in self.config])

        new_image = np.zeros((new_image_height, new_image_width), dtype=image.dtype)

        new_image[0 : chip_size[0], 0 : chip_size[1]] = data["chip2"]
        new_image[
            self.config["chip1"]["yoffset"] : chip_size[0] + self.config["chip1"]["yoffset"],
            chip_size[1] + self.config["chip1"]["xoffset"] : 2 * chip_size[1] + self.config["chip1"]["xoffset"],
        ] = data["chip1"]
        new_image[
            chip_size[0] + self.config["chip3"]["yoffset"] : 2 * chip_size[0] + self.config["chip3"]["yoffset"],
            self.config["chip3"]["xoffset"] : chip_size[1] + self.config["chip3"]["xoffset"],
        ] = data["chip3"]
        new_image[
            chip_size[0] + self.config["chip4"]["yoffset"] : 2 * chip_size[0] + self.config["chip4"]["yoffset"],
            chip_size[1] + self.config["chip4"]["xoffset"] : 2 * chip_size[1] + self.config["chip4"]["xoffset"],
        ] = data["chip4"]

        self.correct_between_chips_1_and_2(new_image)
        self.correct_between_chips_2_and_3(new_image)
        self.correct_between_chips_1_and_4(new_image)
        self.correct_between_chips_3_and_4(new_image)

        return new_image

    def correct_between_chips_1_and_2(self, new_image):
        # between chips 1 and 2, we need to correct the gap
        # gap is config['chip1']['xoffset'] (horizontal) and config['chip1']['yoffset'] (vertical)
        # y will go from 0 to chip_size[0] + config['chip1']['yoffset']
        config = self.config

        for _y in range(0, chip_size[0] + config["chip1"]["yoffset"]):
            left_value = new_image[_y, chip_size[0] - 1]
            right_value = new_image[_y, chip_size[0] + config["chip1"]["xoffset"]]

            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(config["chip1"]["xoffset"])
            if left_value == 0:
                list_new_value = np.ones(config["chip1"]["xoffset"]) * right_value
            elif right_value == 0:
                list_new_value = np.ones(config["chip1"]["xoffset"]) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, config["chip1"]["xoffset"] + 1),
                    [0, config["chip1"]["xoffset"] + 1],
                    [left_value, right_value],
                )

            new_image[_y, chip_size[1] : chip_size[1] + config["chip1"]["xoffset"]] = list_new_value

    def correct_between_chips_2_and_3(self, new_image):
        # gap is config['chip3']['xoffset'] (horizontal) and config['chip3']['yoffset'] (vertical)
        # x will go from 0 to chip_size[1] + config['chip3']['xoffset']
        config = self.config
        for _x in range(0, chip_size[1] + config["chip3"]["xoffset"]):
            left_value = new_image[chip_size[0] - 1, _x]
            right_value = new_image[chip_size[0] + config["chip3"]["yoffset"], _x]

            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(config["chip3"]["yoffset"])
            if left_value == 0:
                list_new_value = np.ones(config["chip3"]["yoffset"]) * right_value
            elif right_value == 0:
                list_new_value = np.ones(config["chip3"]["yoffset"]) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, config["chip3"]["yoffset"] + 1),
                    [0, config["chip3"]["yoffset"] + 1],
                    [left_value, right_value],
                )

            new_image[chip_size[0] : chip_size[0] + config["chip3"]["yoffset"], _x] = list_new_value

    def correct_between_chips_1_and_4(self, new_image):
        # gap is config['chip4']['xoffset'] - config['chip1']['xoffset'] (horizontal) and
        # config['chip4']['yoffset'] - config['chip1']['yoffset'] (vertical)
        # x will go from chip_size[1]+config['chip1']['xoffset'] to 2*chip_size[1]+config['chip1']['xoffset']
        config = self.config
        for _x in range(chip_size[1] + config["chip1"]["xoffset"], 2 * chip_size[1] + config["chip1"]["xoffset"]):
            left_value = new_image[chip_size[0] - 1, _x]
            right_value = new_image[chip_size[0] + config["chip4"]["yoffset"], _x]
            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(config["chip4"]["yoffset"])
            if left_value == 0:
                list_new_value = np.ones(config["chip4"]["yoffset"]) * right_value
            elif right_value == 0:
                list_new_value = np.ones(config["chip4"]["yoffset"]) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, config["chip4"]["yoffset"] + 1),
                    [0, config["chip4"]["yoffset"] + 1],
                    [left_value, right_value],
                )

            new_image[chip_size[0] : chip_size[0] + config["chip4"]["yoffset"], _x] = list_new_value

    def correct_between_chips_3_and_4(self, new_image):
        # gap is config['chip4']['xoffset'] - config['chip3']['xoffset'] (horizontal) and
        # config['chip4']['yoffset'] - config['chip3']['yoffset'] (vertical)
        # y will go from chip_size[0]+config['chip3']['yoffset'] to 2*chip_size[0]+config['chip3']['yoffset']
        config = self.config
        for _y in range(chip_size[0] + config["chip3"]["yoffset"], 2 * chip_size[0] + config["chip3"]["yoffset"]):
            left_value = new_image[_y, chip_size[1] - 1]
            right_value = new_image[_y, chip_size[1] + config["chip4"]["xoffset"]]
            if left_value == 0 and right_value == 0:
                list_new_value = np.zeros(config["chip4"]["xoffset"])
            if left_value == 0:
                list_new_value = np.ones(config["chip4"]["xoffset"]) * right_value
            elif right_value == 0:
                list_new_value = np.ones(config["chip4"]["xoffset"]) * left_value
            else:
                list_new_value = np.interp(
                    np.arange(1, config["chip4"]["xoffset"] + 1),
                    [0, config["chip4"]["xoffset"] + 1],
                    [left_value, right_value],
                )

            new_image[_y, chip_size[1] : chip_size[1] + config["chip4"]["xoffset"]] = list_new_value


if __name__ == "__main__":
    o_corrector = TimepixGeometryCorrection(raw_image_path="notebooks/data/siemens_star.tif")
    corrected = o_corrector.correct()
