import unittest

from cv import zhuocv3 as zc
from cv.image_util import ImageProcessError, preprocess_img


class CVTest(unittest.TestCase):
    def setUp(self) -> None:
        with open('./cv_good_frame.jpeg', 'rb') as img_file:
            self.good_img = img_file.read()

        with open('./cv_bad_frame.jpeg', 'rb') as img_file:
            self.bad_img = img_file.read()

    def test_raw2cv_img(self):
        cv_img = zc.raw2cv_image(self.good_img)

    def test_preprocess_img(self):
        cv_img_good = zc.raw2cv_image(self.good_img)
        cv_img_bad = zc.raw2cv_image(self.bad_img)

        bitmap_good = preprocess_img(cv_img_good)

        with self.assertRaises(expected_exception=ImageProcessError) as e:
            bitmap_bad = preprocess_img(cv_img_bad)

