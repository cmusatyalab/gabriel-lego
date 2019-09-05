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

    def test_init_guidance(self):
        guidance = self.task.get_guidance()

        self.assertTrue(guidance.success)
        self.assertEqual(guidance.step_id, 0)
        self.logger.debug(guidance.instruction)

        # show guidance in Matplotlib
        b, g, r = cv2.split(guidance.image)
        frame_rgb = cv2.merge((r, g, b))
        plt.imshow(frame_rgb)
        plt.title('GuidanceDebug')
        plt.show()

    def test_task_step_completion(self):
        for i, state in enumerate(task_Turtle.bitmaps):
            self.task.update_state(state.copy())
            guidance = self.task.get_guidance()
            self.logger.debug(guidance.instruction)
            self.assertTrue(guidance.success)
            self.assertEqual(i + 1, guidance.step_id)

            # show guidance in Matplotlib
            b, g, r = cv2.split(guidance.image)
            frame_rgb = cv2.merge((r, g, b))
            plt.imshow(frame_rgb)
            plt.title('GuidanceDebug')
            plt.show()
