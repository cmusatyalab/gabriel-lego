import unittest

import numpy as np

from cv import bitmap as bm, zhuocv3 as zc
from cv.image_util import ImageProcessError, preprocess_img


class CVTest(unittest.TestCase):
    correct_state = np.array([[0, 0, 1, 6],
                              [2, 2, 2, 2]])
    incorrect_state = np.array([[0, 2, 1, 1],
                                [0, 2, 1, 6],
                                [2, 2, 2, 2]])

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
        self.assertTrue(bm.bitmap_same(bitmap_good, self.correct_state))
        self.assertFalse(bm.bitmap_same(bitmap_good, self.incorrect_state))

        with self.assertRaises(expected_exception=ImageProcessError) as e:
            bitmap_bad = preprocess_img(cv_img_bad)
