import logging
import random
import sys
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gabriel_lego.lego_engine import tasks
from gabriel_lego.lego_engine.board import BoardState, EmptyBoardState
from gabriel_lego.lego_engine.task_manager import CorrectTaskState, \
    FinalTaskState, IncorrectTaskState, InitialTaskState, NoStateChangeError, \
    _TaskManager


# import cv2


class TaskTest(unittest.TestCase):
    wrong_bitmap = np.array([[0, 4, 6, 4, 4],
                             [0, 4, 4, 4, 4],
                             [0, 5, 5, 5, 5],
                             [0, 5, 0, 0, 5],
                             [5, 5, 0, 5, 5]])

    other_wrong_bitmap = np.array([[4, 6, 4, 4],
                                   [4, 4, 4, 4],
                                   [5, 5, 5, 5],
                                   [5, 0, 0, 5]])

    def setUp(self) -> None:
        self.task = _TaskManager().get_task('turtle_head')
        self.raw_bitmaps = [b.copy()
                            for b in tasks.task_collection['turtle_head']]

        self.logger = logging.getLogger('TestDebug')
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    @staticmethod
    def show_guidance_plt(image, title='GuidanceDebug'):
        # show guidance in Matplotlib
        b, g, r = cv2.split(image)
        frame_rgb = cv2.merge((r, g, b))
        plt.imshow(frame_rgb)
        plt.title(title)
        plt.show()

    def test_initial_state(self):
        state = InitialTaskState(self.task)
        self.logger.debug(state.current_instruction)

        with self.assertRaises(NoStateChangeError):
            state.compute_next_task_state(BoardState(self.raw_bitmaps[2]))

        state = state.compute_next_task_state(EmptyBoardState())

        # make mistake
        state = state.compute_next_task_state(BoardState(self.wrong_bitmap))
        self.assertIsInstance(state, IncorrectTaskState)

        # fix error
        state = state.compute_next_task_state(EmptyBoardState())
        self.assertIsInstance(state, CorrectTaskState)

    def test_complete_correct_run(self,
                                  yield_after_step: bool = False):
        state = InitialTaskState(self.task)
        with self.assertRaises(RuntimeError):
            state.get_current_board_state()
        self.logger.debug(state.current_instruction)

        state = state.compute_next_task_state(EmptyBoardState())
        self.assertEqual(state.get_current_board_state(), EmptyBoardState())
        self.logger.debug(state.current_instruction)
        TaskTest.show_guidance_plt(state.current_image,
                                   title=state.current_instruction)

        for bm in self.raw_bitmaps:
            state = state.compute_next_task_state(BoardState(bm))
            self.assertEqual(state.get_current_board_state(), BoardState(bm))
            self.logger.debug(state.current_instruction)
            TaskTest.show_guidance_plt(state.current_image,
                                       title=state.current_instruction)

            # sending same state multiple times, or an empty board, does
            # nothing (apart from raising an exception...)
            for i in range(random.randint(20, 100)):
                with self.assertRaises(NoStateChangeError):
                    if i % 2 == 0:
                        state.compute_next_task_state(BoardState(bm))
                    else:
                        state.compute_next_task_state(EmptyBoardState())

        self.assertIsInstance(state, FinalTaskState)

    def test_error_recovery(self):
        state = InitialTaskState(self.task) \
            .compute_next_task_state(EmptyBoardState())

        for bm in self.raw_bitmaps[:-1]:
            state = state.compute_next_task_state(BoardState(bm))
            self.assertEqual(state.get_current_board_state(), BoardState(bm))

            # inject error
            state = state.compute_next_task_state(BoardState(self.wrong_bitmap))
            self.assertIsInstance(state, IncorrectTaskState)

            # check that sending the same bitmap does not change the state
            for i in range(random.randint(20, 100)):
                with self.assertRaises(NoStateChangeError):
                    state.compute_next_task_state(BoardState(self.wrong_bitmap))

            # changing to another mistake
            state = state.compute_next_task_state(
                BoardState(self.other_wrong_bitmap))

            self.assertIsInstance(state, IncorrectTaskState)

            # check that sending the same bitmap does not change the state
            for i in range(random.randint(20, 100)):
                with self.assertRaises(NoStateChangeError):
                    state.compute_next_task_state(
                        BoardState(self.other_wrong_bitmap))

            # finally, resolve error
            state = state.compute_next_task_state(BoardState(bm))
            self.assertEqual(state.get_current_board_state(), BoardState(bm))
            self.assertIsInstance(state, CorrectTaskState)
