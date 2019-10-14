import time

from gabriel_lego.cv import bitmap as bm
from gabriel_lego.cv.colors import *
from gabriel_lego.cv.image_util import preprocess_img

white = LEGOColorWhite
green = LEGOColorGreen
yellow = LEGOColorYellow
red = LEGOColorRed
blue = LEGOColorBlue
black = LEGOColorBlack

expected_bitmap = np.array(
    [[green.mapping] * 4,
     [blue.mapping] * 4,
     [red.mapping] * 4,
     [yellow.mapping] * 4,
     [black.mapping] * 4,
     [white.mapping] * 4])

image_path = './test_frames/green_blue_red_yellow_black_white.jpeg'


def _execute(img: np.ndarray) -> float:
    ti = time.time()
    bitmap = preprocess_img(img)
    assert bm.bitmap_same(bitmap, expected_bitmap)
    return time.time() - ti


def _process(reps: int):
    img = cv2.imread(image_path)
    print(min([_execute(img) for _ in range(reps)]))


if __name__ == '__main__':
    _process(100)
