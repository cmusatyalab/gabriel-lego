from __future__ import annotations

from enum import IntEnum
from typing import Dict, Tuple

import cv2
import numpy as np


class NoSuchColorError(Exception):
    pass


# Color mappings
# nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6, unsure:7
class LEGOColorID(IntEnum):
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


class HSVRange:
    def __init__(self, low_bound: HSVValue, high_bound: HSVValue):
        super().__init__()
        self._low_bound = low_bound
        self._high_bound = high_bound

    def __eq__(self, other: HSVRange) -> bool:
        return self._low_bound == other._low_bound \
               and self._high_bound == other._high_bound

    @property
    def low_bound(self) -> HSVValue:
        return self._low_bound

    @property
    def high_bound(self) -> HSVValue:
        return self._high_bound

    def get_mask(self, img) -> np.ndarray:
        return cv2.inRange(img,
                           self._low_bound.to_cv2_HSV(),
                           self._high_bound.to_cv2_HSV())


class RawCVColor:
    def __init__(self, ranges: Tuple[HSVRange]):
        super().__init__()
        self._ranges = ranges

    @property
    def ranges(self) -> Tuple[HSVRange]:
        return self._ranges

    def __eq__(self, other: RawCVColor):
        return all(ours == theirs for ours, theirs
                   in zip(self.ranges, other.ranges)) \
            if len(self.ranges) == len(other.ranges) else False

    def get_cv2_masks(self,
                      hsv_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # src_mask = np.ones(hsv_img.shape[0:2], dtype=np.uint8)

        raw_mask = self._ranges[0].get_mask(hsv_img)
        for range in self._ranges[1:]:
            raw_mask = cv2.bitwise_or(raw_mask, range.get_mask(hsv_img))

        mask = cv2.bitwise_and(hsv_img, hsv_img, mask=raw_mask)
        mask_bool = mask.astype(bool)

        return mask, mask_bool


class LEGOCVColor(RawCVColor):
    range_bounds: Dict[LEGOColorID, Tuple[HSVRange]] = {
        LEGOColorID.WHITE : (HSVRange(low_bound=HSVValue(0, 0, 60),
                                      high_bound=HSVValue(359, 60, 100)),),
        ###
        LEGOColorID.GREEN : (HSVRange(low_bound=HSVValue(90, 50, 20),
                                      high_bound=HSVValue(160, 100, 100)),),
        ###
        LEGOColorID.YELLOW: (HSVRange(low_bound=HSVValue(30, 50, 50),
                                      high_bound=HSVValue(60, 100, 100)),),
        ###
        # red is special, it "wraps around" the hue scale
        LEGOColorID.RED   : (HSVRange(low_bound=HSVValue(330, 50, 50),
                                      high_bound=HSVValue(359, 100, 100)),
                             HSVRange(low_bound=HSVValue(0, 50, 50),
                                      high_bound=HSVValue(20, 100, 100))),
        ###
        LEGOColorID.BLUE  : (HSVRange(low_bound=HSVValue(200, 50, 20),
                                      high_bound=HSVValue(270, 100, 100)),),
        ###
        LEGOColorID.BLACK : (HSVRange(low_bound=HSVValue(0, 0, 0),
                                      high_bound=HSVValue(359, 50, 50)),)
    }

    def __init__(self, color_id: LEGOColorID):
        """
        :param color_id: The numerical ID of this color.
        """
        try:
            super().__init__(ranges=LEGOCVColor.range_bounds[color_id])
        except KeyError:
            raise NoSuchColorError(color_id)

        self._value_mapping: LEGOColorID = color_id

    @property
    def mapping(self) -> LEGOColorID:
        return self._value_mapping


if __name__ == '__main__':
    # debug using one the test frames

    img_path = 'green_blue_red_yellow_black_white.jpeg'
    cv_img = cv2.imread(img_path)

    cv_img = cv2.resize(cv_img, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_CUBIC)

    # cv_img = zc.raw2cv_image(raw_img)

    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV', hsv_img)

    for color_id in LEGOColorID:
        color = LEGOCVColor(color_id)
        mask, _ = color.get_cv2_masks(hsv_img)
        cv2.imshow(f'{color_id.name}', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
