import unittest
from typing import Dict

import numpy as np

from gabriel_lego.cv import bitmap as bm, zhuocv3 as zc
from gabriel_lego.cv.image_util import preprocess_img
from gabriel_lego.cv.lego_cv import LEGOCVError, LEGO_COLORS, \
    NoLEGODetectedError
from gabriel_lego.lego_engine import tasks


def add_block_tests(cls: unittest.TestCase) -> unittest.TestCase:
    '''
    Wrapper that adds individual tests for simple cases consisting of
    recognizing simple combinations of blocks (that way they
    can pass/fail individually, making debugging easier).

    :param cls: TestCase to wrap

    :return: The wrapped TestCase with added test methods
    '''
    ind_blocks = {
        'white' : np.array([[LEGO_COLORS.WHITE.value.mapping] * 4]),
        'green' : np.array([[LEGO_COLORS.GREEN.value.mapping] * 4]),
        'yellow': np.array([[LEGO_COLORS.YELLOW.value.mapping] * 4]),
        'red'   : np.array([[LEGO_COLORS.RED.value.mapping] * 4]),
        'blue'  : np.array([[LEGO_COLORS.BLUE.value.mapping] * 4]),
        'black' : np.array([[LEGO_COLORS.BLACK.value.mapping] * 4])  # ACAB
    }

    block_combinations = {
        'black_blue_green'                 :
            np.array(
                [[LEGO_COLORS.BLACK.value.mapping * 4],
                 [LEGO_COLORS.BLUE.value.mapping * 4],
                 [LEGO_COLORS.GREEN.value.mapping * 4]]),
        'green_blue_red_yellow_black_white':
            np.array(
                [[LEGO_COLORS.GREEN.value.mapping * 4],
                 [LEGO_COLORS.BLUE.value.mapping * 4],
                 [LEGO_COLORS.RED.value.mapping * 4],
                 [LEGO_COLORS.YELLOW.value.mapping * 4],
                 [LEGO_COLORS.BLACK.value.mapping * 4],
                 [LEGO_COLORS.WHITE.value.mapping * 4]]),
        'yellow_white'                     :
            np.array(
                [[LEGO_COLORS.YELLOW.value.mapping * 4],
                 [LEGO_COLORS.WHITE.value.mapping * 4]])

    }

    def gen_test_case(raw_img: bytes, exp_bitmap: np.ndarray):
        def _fn(self: unittest.TestCase):
            parsed_bitmap = preprocess_img(zc.raw2cv_image(raw_img))
            self.assertTrue(bm.bitmap_same(exp_bitmap, parsed_bitmap),
                            msg=f'\nExpected bitmap: \n{exp_bitmap}\n'
                                f'\nReceived bitmap: \n{parsed_bitmap}\n')

        return _fn

    for color, exp_bitmap in ind_blocks.items():
        with open(f'./test_frames/{color}.jpeg', 'rb') as f:
            method = gen_test_case(f.read(), exp_bitmap)

        setattr(cls, f'test_{color}_block', method)

    for combination, exp_bitmap in block_combinations.items():
        with open(f'./test_frames/{combination}.jpeg', 'rb') as f:
            method = gen_test_case(f.read(), exp_bitmap)

        setattr(cls, f'test_{combination}_combination', method)

    return cls


@add_block_tests
class CVTest(unittest.TestCase):
    correct_state = np.array([[0, 0, 1, 6],
                              [2, 2, 2, 2]])
    incorrect_state = np.array([[0, 0, 1, 1],
                                [0, 0, 1, 6],
                                [2, 2, 2, 2]])

    # Simple, individual block and color tests
    # nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6, unsure:7

    def setUp(self) -> None:
        with open('./cv_good_frame.jpeg', 'rb') as img_file:
            self.good_img = img_file.read()

        with open('./cv_bad_frame.jpeg', 'rb') as img_file:
            self.bad_img = img_file.read()

        self.step_frames_1 = []
        self.step_frames_2 = []
        for i in range(8):
            with open(f'./frames/step_{i}.jpeg', 'rb') as f:
                self.step_frames_1.append(f.read())

            with open(f'./frames_2/step_{i}.jpeg', 'rb') as f:
                self.step_frames_2.append(f.read())

        self.task_bitmaps = tasks.task_collection['turtle_head']

    def test_cv_real_frames(self):
        # clear, beautiful frames first:
        # first frame is a pic of an empty board
        frames = [zc.raw2cv_image(frame) for frame in self.step_frames_1]
        with self.assertRaises(NoLEGODetectedError):
            preprocess_img(frames[0])

        for frame, correct_bm in zip(frames[1:], self.task_bitmaps):
            bitmap = preprocess_img(frame)
            self.assertTrue(bm.bitmap_same(bitmap, correct_bm),
                            msg=f'\nCorrect bitmap: \n{correct_bm}\n'
                                f'\nReceived bitmap: \n{bitmap}\n')

        # blurry frames second
        # first frame is a pic of an empty board
        # frames = [zc.raw2cv_image(frame) for frame in self.step_frames_2]
        # with self.assertRaises(NoLEGODetectedError):
        #     preprocess_img(frames[0])
        #
        # for frame, correct_bm in zip(frames[1:], self.task_bitmaps):
        #     bitmap = preprocess_img(frame)
        #     self.assertTrue(bm.bitmap_same(bitmap, correct_bm),
        #                     msg=f'\nCorrect bitmap: \n{correct_bm}\n'
        #                         f'\nReceived bitmap: \n{bitmap}\n')

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
