import unittest
from typing import Dict

import numpy as np

import bitmap as bm
import zhuocv3 as zc
from image_util import preprocess_img
from lego_cv import LEGOCVError


class CVTest(unittest.TestCase):
    correct_state = np.array([[0, 0, 1, 6],
                              [2, 2, 2, 2]])
    incorrect_state = np.array([[0, 0, 1, 1],
                                [0, 0, 1, 6],
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

        with self.assertRaises(expected_exception=LEGOCVError) as e:
            bitmap_bad = preprocess_img(cv_img_bad)

    def test_bitmaps_diff(self):
        self.assertIsNone(bm.bitmap_diff(self.correct_state,
                                         self.correct_state.copy()))
        self.assertIsNone(bm.bitmap_diff(self.incorrect_state,
                                         self.incorrect_state.copy()))

        diff_1 = bm.bitmap_diff(self.correct_state, self.incorrect_state)
        diff_2 = bm.bitmap_diff(self.incorrect_state, self.correct_state)

        self.assertIsNotNone(diff_1)
        self.assertIsInstance(diff_1, Dict)

        self.assertIsNotNone(diff_2)
        self.assertIsInstance(diff_2, Dict)

        self.assertEqual(diff_1['n_diff_pieces'], 1)
        self.assertEqual(diff_1['larger'], 2)

        self.assertEqual(diff_2['n_diff_pieces'], 1)
        self.assertEqual(diff_2['larger'], 1)

        self.assertListEqual(diff_1['first_piece'], diff_2['first_piece'])
