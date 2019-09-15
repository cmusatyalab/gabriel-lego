from __future__ import annotations

from enum import Enum
from typing import Tuple

import cv2
import numpy as np


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


class LEGOCVColor:
    def __init__(self,
                 value_mapping: int,
                 range_low: HSVValue,
                 range_high: HSVValue):
        """
        :param value_mapping: The numerical representation of this color.

        :param range_low: The lower bound on the HSV range for this color.

        :param range_high: The upper bound on the HSV range for this color.
        """

        self._value_mapping: int = value_mapping
        self._lower_range: HSVValue = range_low
        self._upper_range: HSVValue = range_high

    @property
    def mapping(self) -> int:
        return self._value_mapping

    @property
    def lower_range(self) -> HSVValue:
        return self._lower_range

    @property
    def upper_range(self) -> HSVValue:
        return self._upper_range

    def __eq__(self, other: LEGOCVColor):
        return self.mapping == other.mapping

    def get_cv2_masks(self,
                      hsv_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        src_mask = np.ones(hsv_img.shape[0:2], dtype=np.uint8)

        if self._upper_range.hue < self._lower_range.hue:
            # mask needs to "wrap around" the colorspace, so we'll define two
            # ranges; one for low -> max_colorspace and one for min_colorspace
            # -> high, and build the mask combining the two
            # note that saturation and value DO NOT wrap around

            tmp_upper = HSVValue(hue=359,
                                 saturation=self._upper_range.saturation,
                                 value=self._upper_range.value)

            tmp_lower = HSVValue(hue=0,
                                 saturation=self._lower_range.saturation,
                                 value=self._lower_range.value)

            mask_1 = cv2.inRange(hsv_img,
                                 self._lower_range.to_cv2_HSV(),
                                 tmp_upper.to_cv2_HSV())

            mask_2 = cv2.inRange(hsv_img,
                                 tmp_lower.to_cv2_HSV(),
                                 self._upper_range.to_cv2_HSV())

            raw_mask = cv2.bitwise_or(mask_1, mask_2)

        else:
            raw_mask = cv2.inRange(hsv_img,
                                   self.lower_range.to_cv2_HSV(),
                                   self.upper_range.to_cv2_HSV())

        mask = cv2.bitwise_and(raw_mask, src_mask)
        mask_bool = mask.astype(bool)

        return mask, mask_bool


# Color mappings
# nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6, unsure:7
class LEGOColors(LEGOCVColor, Enum):
    WHITE = LEGOCVColor(1)
    GREEN = LEGOCVColor(value_mapping=2,
                        range_low=HSVValue(80, 50, 50),
                        range_high=HSVValue(160, 50, 50))
    YELLOW = LEGOCVColor(3)
    RED = LEGOCVColor(value_mapping=4,
                      range_low=HSVValue(330, 50, 50),
                      range_high=HSVValue(20, 50, 50))
    BLUE = LEGOCVColor(value_mapping=5,
                       range_low=HSVValue(200, 50, 50),
                       range_high=HSVValue(270, 50, 50))
    BLACK = LEGOCVColor(6)

    # TODO: use them in the code
