import cv2
import numpy as np

from gabriel_lego.cv import lego_cv as lc
from gabriel_lego.lego_engine import config


def resize_img(img):
    return img if img.shape == (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3) \
        else cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
                        interpolation=cv2.INTER_AREA)


def preprocess_img(img) -> np.ndarray:
    stretch_ratio = float(16) / 9 * img.shape[0] / img.shape[1]
    img = resize_img(img)

    #
    # zc.check_and_display('input', img, config.DISPLAY_LIST,
    #                      wait_time=config.DISPLAY_WAIT_TIME,
    #                      resize_max=config.DISPLAY_MAX_PIXEL)

    bitmap = lc.process(img, stretch_ratio, config.DISPLAY_LIST)
    return bitmap
