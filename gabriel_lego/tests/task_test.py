import logging
import random
import sys
import unittest

import cv2
import matplotlib.pyplot as plt

from gabriel_lego.lego_engine.task_manager import BoardState, EmptyBoardState, \
    TaskManager
# import cv2
from gabriel_lego.lego_engine.tasks import task_collection


class TaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskManager()
        self.task_name = random.choice(list(task_collection.keys()))
        self.task_states = [BoardState(b)
                            for b in task_collection[self.task_name]]

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

    def test_initial_guidance(self, display: bool = False):
        guidance = self.task.get_guidance(
            task_name=self.task_name,
            board_state=EmptyBoardState(),
            target_state_index=-1,
            previous_state_index=-1,
            prev_timestamp=0
        )

        self.assertTrue(guidance.success)
        self.assertEqual(guidance.target_state_index, 0)
        self.assertEqual(guidance.previous_state_index, -1)
        self.assertFalse(guidance.task_finished)
        if display:
            TaskTest.show_guidance_plt(guidance.image, guidance.instruction)

    def test_correct_step(self):
        state_index, state = random.choice(
            [(i, state) for i, state in enumerate(self.task_states[:-1])]
        )  # choose a random state in the task, that's not the last step

        guidance = self.task.get_guidance(
            task_name=self.task_name,
            board_state=state,
            target_state_index=state_index,
            previous_state_index=-1,
            prev_timestamp=0
        )

        self.assertTrue(guidance.success, f'{guidance}')
        self.assertEqual(guidance.target_state_index, state_index + 1,
                         f'{guidance}')
        self.assertEqual(guidance.previous_state_index, state_index,
                         f'{guidance}')

    def test_incorrect_step(self):
        state_index, state = random.choice(
            [(i, state) for i, state in enumerate(self.task_states[:-1])
             if i != 0]
        )  # choose a random state in the task, that's not the last step or
        # the first one, as those have special behavior

        previous_state_index = state_index - 1

        guidance = self.task.get_guidance(
            task_name=self.task_name,
            board_state=state,
            target_state_index=state_index + 1,
            previous_state_index=previous_state_index,
            prev_timestamp=0
        )

        self.assertFalse(guidance.success, f'{guidance}')
        self.assertEqual(guidance.target_state_index, state_index - 1,
                         f'{guidance}')
        self.assertEqual(guidance.previous_state_index, -1, f'{guidance}')

    def test_final_step(self, display: bool = False):
        guidance = self.task.get_guidance(
            task_name=self.task_name,
            board_state=self.task_states[-1],
            target_state_index=len(self.task_states) - 1,
            previous_state_index=len(self.task_states) - 2,
            prev_timestamp=0
        )

        self.assertTrue(guidance.success, f'{guidance}')
        self.assertEqual(guidance.target_state_index, -1, f'{guidance}')
        self.assertEqual(guidance.previous_state_index, -1, f'{guidance}')
        self.assertTrue(guidance.task_finished, f'{guidance}')

        if display:
            TaskTest.show_guidance_plt(guidance.image, guidance.instruction)

    def test_complete_correct_run(self, yield_after_step: bool = False):
        # perfect run
        # first, test initial guidance
        self.test_initial_guidance(display=True)
        if yield_after_step:
            yield

        # iterate over states (except last one)
        for i, state in enumerate(self.task_states[:-1]):
            guidance = self.task.get_guidance(
                task_name=self.task_name,
                board_state=state,
                target_state_index=i,
                previous_state_index=i - 1,
                prev_timestamp=0
            )

            self.assertTrue(guidance.success)
            self.assertEqual(guidance.target_state_index, i + 1)
            self.assertFalse(guidance.task_finished)
            TaskTest.show_guidance_plt(guidance.image, guidance.instruction)

            if yield_after_step:
                yield

        # final step
        self.test_final_step(display=True)
        if yield_after_step:
            yield

    def test_intercalated_independent_correct_runs(self):
        # basically to test the statelessness of the implementation
        for _, _ in zip(self.test_complete_correct_run(yield_after_step=True),
                        self.test_complete_correct_run(yield_after_step=True)):
            pass

    def test_error_recovery(self):
        state_index, state = random.choice(
            [(i, state) for i, state in enumerate(self.task_states[:-1])
             if i != 0]
        )  # choose a random state in the task, that's not the last step or
        # the first one, as those have special behavior

        guidance = self.task.get_guidance(
            task_name=self.task_name,
            board_state=state,
            target_state_index=state_index,
            previous_state_index=-1,
            prev_timestamp=0
        )

        self.assertTrue(guidance.success, f'{guidance}')
        self.assertEqual(guidance.target_state_index, state_index + 1,
                         f'{guidance}')
        self.assertEqual(guidance.previous_state_index, state_index,
                         f'{guidance}')

        # now send a wrong step
        previous_state_index = guidance.previous_state_index
        state_index, state = random.choice(
            [(i, state) for i, state in enumerate(self.task_states[:-1])
             if i != 0 and i != state_index + 1 and i != state_index]
        )  # choose a random state in the task, that's not the last step or
        # the first one, or the one we SHOULD've sent or the one we already sent

        guidance = self.task.get_guidance(
            task_name=self.task_name,
            board_state=state,
            target_state_index=state_index + 1,
            previous_state_index=previous_state_index,
            prev_timestamp=0
        )

        self.assertFalse(guidance.success, f'{guidance}')
        self.assertEqual(guidance.target_state_index, previous_state_index,
                         f'{guidance}')
        self.assertEqual(guidance.previous_state_index, -1, f'{guidance}')

        TaskTest.show_guidance_plt(guidance.image, guidance.instruction)

        # now recover from the error
        guidance = self.task.get_guidance(
            task_name=self.task_name,
            board_state=self.task_states[guidance.target_state_index],
            target_state_index=guidance.target_state_index,
            previous_state_index=guidance.previous_state_index,
            prev_timestamp=0
        )

        self.assertTrue(guidance.success, f'{guidance}')

        TaskTest.show_guidance_plt(guidance.image, guidance.instruction)
