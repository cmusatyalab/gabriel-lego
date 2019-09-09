import logging
import random
import sys
import unittest

import cv2
import matplotlib.pyplot as plt

from lego_engine.tasks import Task, task_Turtle
# import cv2
from lego_engine.tasks.Task import BoardState, EmptyBoardState, \
    NoGuidanceError, \
    NoStateChangeError


class TaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = Task(task_Turtle.bitmaps)
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

    def __update_and_check_state(self, state: BoardState,
                                 index: int, success=True):
        self.task.update_state(state)
        guidance = self.task.get_guidance()
        self.logger.debug(guidance.instruction)
        self.assertEqual(guidance.success, success)
        self.assertEqual(index, guidance.step_id)

        if success:
            TaskTest.show_guidance_plt(guidance.image)
        else:
            if guidance.image is not None:
                TaskTest.show_guidance_plt(guidance.image,
                                           title='GuidanceDebug_ExpectedError')

    def test_init_guidance(self):
        with self.assertRaises(NoGuidanceError):
            # getting guidance without first sending a state should cause an
            # error
            self.task.get_guidance()

        self.task.update_state(EmptyBoardState())
        guidance = self.task.get_guidance()

        self.assertTrue(guidance.success)
        self.assertEqual(guidance.step_id, 0)
        self.logger.debug(guidance.instruction)

        TaskTest.show_guidance_plt(guidance.image)

    def test_task_step_completion(self):
        # check initial guidance using empty board
        self.__update_and_check_state(EmptyBoardState(), 0)

        for i, bitmap in enumerate(task_Turtle.bitmaps):
            state = BoardState(bitmap.copy())
            self.__update_and_check_state(state, i + 1)

    def test_same_state_error(self):
        # check initial guidance using empty board
        self.__update_and_check_state(EmptyBoardState(), 0)

        for i, bitmap in enumerate(task_Turtle.bitmaps[:2]):
            state = BoardState(bitmap.copy())
            self.__update_and_check_state(state, i + 1)

        with self.assertRaises(NoStateChangeError):
            state = BoardState(bitmap.copy())
            self.__update_and_check_state(state, i + 1)

    def test_task_step_error(self):
        # check initial guidance using empty board
        self.__update_and_check_state(EmptyBoardState(), 0)

        for i, bitmap in enumerate(task_Turtle.bitmaps[:2]):
            state = BoardState(bitmap.copy())
            self.__update_and_check_state(state, i + 1)

        self.__update_and_check_state(
            state=BoardState(task_Turtle.bitmaps[-1].copy()),
            index=-1,
            success=False
        )

    def test_pass_empty_board_error(self):
        # check initial guidance using empty board
        self.__update_and_check_state(EmptyBoardState(), 0)

        for i, bitmap in enumerate(task_Turtle.bitmaps[:2]):
            state = BoardState(bitmap.copy())
            self.__update_and_check_state(state, i + 1)

        self.__update_and_check_state(
            state=EmptyBoardState(),
            index=-1,
            success=False
        )

    def test_wrong_initial_state(self):
        # send a state instead of the empty board as initial message
        # should cause an error
        self.__update_and_check_state(
            BoardState(random.choice(task_Turtle.bitmaps).copy()),
            -1,
            success=False)

    def test_error_recovery(self):
        self.__update_and_check_state(EmptyBoardState(), 0)

        # alternate steps and errors to check error recovery
        for i, frame in enumerate(task_Turtle.bitmaps[:-1]):
            self.__update_and_check_state(BoardState(frame.copy()), i + 1)

            wrong_frame_idx = [j for j in range(len(task_Turtle.bitmaps))
                               if j != i + 1 and j != i]
            wrong_frame = task_Turtle.bitmaps[random.choice(wrong_frame_idx)]

            self.__update_and_check_state(BoardState(wrong_frame.copy()), -1,
                                          success=False)

            self.__update_and_check_state(BoardState(frame.copy()), i + 1)
