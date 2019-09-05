import logging
import sys
import unittest

import cv2
import matplotlib.pyplot as plt

from lego_engine.tasks import Task, task_Turtle


# import cv2


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

    def __update_and_check_state(self, state, index, success=True):
        self.task.update_state(state.copy())
        guidance = self.task.get_guidance()
        self.logger.debug(guidance.instruction)
        self.assertEqual(guidance.success, success)
        self.assertEqual(index, guidance.step_id)

        if success:
            TaskTest.show_guidance_plt(guidance.image)
        else:
            TaskTest.show_guidance_plt(guidance.image,
                                       title='GuidanceDebug_ExpectedError')

    def test_init_guidance(self):
        guidance = self.task.get_guidance()

        self.assertTrue(guidance.success)
        self.assertEqual(guidance.step_id, 0)
        self.logger.debug(guidance.instruction)

        TaskTest.show_guidance_plt(guidance.image)

    def test_task_step_completion(self):
        for i, state in enumerate(task_Turtle.bitmaps):
            self.__update_and_check_state(state, i + 1)

    def test_task_step_error(self):
        for i, state in enumerate(task_Turtle.bitmaps[:2]):
            self.__update_and_check_state(state, i + 1)

        self.__update_and_check_state(
            state=task_Turtle.bitmaps[-1].copy(),
            index=-1,
            success=False
        )
