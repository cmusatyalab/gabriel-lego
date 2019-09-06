import random
import time
from typing import NamedTuple, Optional

import numpy as np

from cv import bitmap as bm
from lego_engine import config


class Guidance(NamedTuple):
    success: bool
    instruction: str
    image: Optional[np.ndarray]
    step_id: int


class Task:
    def __init__(self, bitmaps):
        self.states = bitmaps

        self.prev_time = None
        self.current_time = time.time()
        self.good_word_idx = 0

        self.prev_good_state_idx = 0
        # initial guidance
        self.target_state_idx = 0
        target_state = self.states[self.target_state_idx]

        instruction = \
            "Welcome to the Lego task. As a first step, please " \
            "find a piece of 1x%d %s brick and put it on the " \
            "board." % (target_state.shape[1],
                        config.COLOR_ORDER[target_state[0, 0]])
        # animation = bm.bitmap2guidance_animation(target_state,
        #                                         config.ACTION_TARGET)
        img_guidance = bm.bitmap2guidance_img(target_state, None,
                                              config.ACTION_TARGET)
        self.current_guidance = Guidance(
            success=True,
            instruction=instruction,
            image=img_guidance,
            step_id=0
        )

    def update_state(self, bitmap) -> None:
        new_state = bitmap
        self.prev_time = self.current_time
        self.current_time = time.time()

        target_state = self.states[self.target_state_idx]

        # Check if we at least reached the previously desired state
        if bm.bitmap_same(new_state, target_state):

            if self.target_state_idx == (len(self.states) - 1):
                # Task is done
                instruction = "You have completed the task. " \
                              "Congratulations!"
                # result['animation'] = bm.bitmap2guidance_animation(
                #     self.current_state, config.ACTION_TARGET)
                img_guidance = bm.bitmap2guidance_img(new_state, None,
                                                      config.ACTION_TARGET)
                self.current_guidance = Guidance(
                    success=True,
                    instruction=instruction,
                    image=img_guidance,
                    step_id=self.target_state_idx + 1
                )
                return

            # Not done
            # Next state is simply the next one in line
            self.prev_good_state_idx = self.target_state_idx
            self.target_state_idx += 1
            target_state = self.states[self.target_state_idx]

            # Determine the type of change needed for the next step

            diff = bm.bitmap_diff(new_state, target_state)
            assert diff  # states can't be the same
            assert diff['n_diff_pieces'] == 1  # for now only change one
            # piece at the time

            self.good_word_idx = (self.good_word_idx +
                                  random.randint(1, 3)) % 4

            if diff['larger'] == 2:  # target state has one more piece
                instruction = bm.generate_message(
                    new_state,
                    target_state,
                    config.ACTION_ADD,
                    diff['first_piece'],
                    step_time=self.current_time - self.prev_time,
                    good_word_idx=self.good_word_idx)

                # result['animation'] = bm.bitmap2guidance_animation(
                #    target_state,
                #    config.ACTION_ADD,
                #    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(target_state,
                                                      diff['first_piece'],
                                                      config.ACTION_ADD)
            else:  # target state has one less piece
                instruction = bm.generate_message(
                    new_state,
                    target_state,
                    config.ACTION_REMOVE,
                    diff['first_piece'],
                    step_time=self.current_time - self.prev_time,
                    good_word_idx=self.good_word_idx)

                # result['animation'] = bm.bitmap2guidance_animation(
                #    self.current_state,
                #     config.ACTION_REMOVE,
                #    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(new_state,
                                                      diff['first_piece'],
                                                      config.ACTION_REMOVE)

            self.current_guidance = Guidance(
                success=True,
                instruction=instruction,
                image=img_guidance,
                step_id=self.target_state_idx
            )

        else:
            # reached an erroneous state
            self.target_state_idx = self.prev_good_state_idx
            instruction = "This is incorrect, please undo the last " \
                          "step and revert to the model shown on " \
                          "the screen."
            # result['animation'] = bm.bitmap2guidance_animation(
            #     self.prev_good_state, config.ACTION_TARGET)
            target_state = self.states[self.target_state_idx]
            img_guidance = bm.bitmap2guidance_img(target_state,
                                                  None,
                                                  config.ACTION_TARGET)
            self.current_guidance = Guidance(
                success=False,
                instruction=instruction,
                image=img_guidance,
                step_id=-1
            )

    def get_guidance(self) -> Guidance:
        return self.current_guidance
