from __future__ import annotations

from enum import IntEnum

import cv2
import numpy as np


class NoSuchColorError(Exception):
    pass


# Color mappings
# nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6, unsure:7
class LEGOColorID(IntEnum):
    NOTHING = 0
    WHITE = 1
    GREEN = 2
    YELLOW = 3
    RED = 4
    BLUE = 5
    BLACK = 6


class HSVValue:
    """
    Represents an HSV value using the traditional ranges for hue, saturation
    and value instead of OpenCVs weird ranges. Provides a method to convert
    to a OpenCV compatible np.ndarray though.
    """

    def __init__(self, hue: int, saturation: int, value: int):
        assert hue in range(0, 360)
        assert 0 <= saturation <= 100
        assert 0 <= value <= 100

        self._hue = hue
        self._saturation = saturation
        self._value = value

    def __eq__(self, other: HSVValue) -> bool:
        return self._hue == other._hue \
               and self._saturation == other._saturation \
               and self._value == other._value

    @property
    def hue(self):
        return self._hue

    @property
    def saturation(self):
        return self._saturation

    @property
    def value(self):
        return self._value

    def to_cv2_HSV(self) -> np.ndarray:
        cv_hue = self._hue // 2
        cv_saturation = np.floor(255 * (self._saturation / 100))
        cv_value = np.floor(255 * (self._value / 100))

        return np.array([cv_hue, cv_saturation, cv_value], dtype=np.uint8)


class SimpleHSVColor:
    def __init__(self,
                 low_bound: HSVValue,
                 high_bound: HSVValue,
                 color_id: LEGOColorID = LEGOColorID.NOTHING):
        super().__init__()
        self._color_id = color_id
        self._low_bound = low_bound
        self._high_bound = high_bound

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SimpleHSVColor):
            return self._low_bound == other._low_bound \
                   and self._high_bound == other._high_bound
        else:
            return False

    @property
    def mapping(self):
        return self._color_id

    @property
    def low_bound(self) -> HSVValue:
        return self._low_bound

    @property
    def high_bound(self) -> HSVValue:
        return self._high_bound

    def get_mask(self, img) -> np.ndarray:
        if self._high_bound.hue < self._low_bound.hue:
            # mask needs to "wrap around" the colorspace, so we'll define two
            # ranges; one for low -> max_colorspace and one for min_colorspace
            # -> high, and build the mask combining the two
            # note that saturation and value DO NOT wrap around

            tmp_upper = HSVValue(hue=359,
                                 saturation=self._high_bound.saturation,
                                 value=self._high_bound.value)

            tmp_lower = HSVValue(hue=0,
                                 saturation=self._low_bound.saturation,
                                 value=self._low_bound.value)

            mask_1 = cv2.inRange(img,
                                 self._low_bound.to_cv2_HSV(),
                                 tmp_upper.to_cv2_HSV())

            mask_2 = cv2.inRange(img,
                                 tmp_lower.to_cv2_HSV(),
                                 self._high_bound.to_cv2_HSV())

            return cv2.bitwise_or(mask_1, mask_2)

        else:
            return cv2.inRange(img,
                               self._low_bound.to_cv2_HSV(),
                               self._high_bound.to_cv2_HSV())


LEGOColorWhite = SimpleHSVColor(low_bound=HSVValue(0, 0, 75),
                                high_bound=HSVValue(359, 24, 100),
                                color_id=LEGOColorID.WHITE)

LEGOColorGreen = SimpleHSVColor(low_bound=HSVValue(90, 30, 20),
                                high_bound=HSVValue(170, 100, 100),
                                color_id=LEGOColorID.GREEN)

LEGOColorYellow = SimpleHSVColor(low_bound=HSVValue(30, 50, 50),
                                 high_bound=HSVValue(60, 100, 100),
                                 color_id=LEGOColorID.YELLOW)

# red is special
LEGOColorRed = SimpleHSVColor(low_bound=HSVValue(310, 40, 40),
                              high_bound=HSVValue(20, 100, 100),
                              color_id=LEGOColorID.RED)
###

LEGOColorBlue = SimpleHSVColor(low_bound=HSVValue(200, 30, 40),
                               high_bound=HSVValue(270, 100, 100),
                               color_id=LEGOColorID.BLUE)

LEGOColorBlack = SimpleHSVColor(low_bound=HSVValue(0, 0, 0),
                                high_bound=HSVValue(359, 30, 50),
                                color_id=LEGOColorID.BLACK)

LEGOColorDOBMaskBlack = SimpleHSVColor(
    low_bound=HSVValue(0, 15, 0),
    high_bound=HSVValue(359, 100, 100),
    color_id=LEGOColorID.NOTHING)

if __name__ == '__main__':
    import gabriel_lego.cv.zhuocv3 as zc
    from gabriel_lego.lego_engine import config

    # debug using one the test frames

    colors = {
        LEGOColorID.WHITE : LEGOColorWhite,
        LEGOColorID.GREEN : LEGOColorGreen,
        LEGOColorID.YELLOW: LEGOColorYellow,
        LEGOColorID.RED   : LEGOColorRed,
        LEGOColorID.BLUE  : LEGOColorBlue,
        LEGOColorID.BLACK : LEGOColorBlack,
    }

    # img_path = 'green_blue_red_yellow_black_white.jpeg'
    img_path = '../tests/test_frames/yellow_white.jpeg'
    cv_img = cv2.imread(img_path)

    cv_img = cv2.resize(cv_img, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_CUBIC)

    # cv_img = zc.raw2cv_image(raw_img)

    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsv_img)

    for color_id in LEGOColorID:
        if color_id == LEGOColorID.NOTHING:
            continue

        color = colors[color_id]
        mask = color.get_mask(hsv_img)
        cv2.imshow(f'{color_id.name}', mask)

    dob = zc.get_DoB(cv_img, config.BLUR_KERNEL_SIZE, 1, method='Average')
    cv2.imshow('blur', dob)

    # dob_hsv = cv2.cvtColor(dob, cv2.COLOR_BGR2HSV)
    dob_mask = LEGOColorDOBMaskBlack.get_mask(dob)
    cv2.imshow('dob_mask', dob_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
