import random
import time
from typing import NamedTuple, Optional

from cv import bitmap as bm
from lego_engine import config


class Guidance(NamedTuple):
    success: bool
    instruction: str
    image: Optional[bytes]
    step_id: int


class Task:
    def __init__(self, bitmaps):
        self.target_state_idx = -1
        self.current_state = None
        self.states = bitmaps
        self.time_estimates = [0] * len(bitmaps)
        self.prev_good_state = self.states[0]
        self.prev_time = None
        self.current_time = time.time()
        self.good_word_idx = 0

    def state2idx(self, state):
        for idx, s in enumerate(self.states):
            if bm.bitmap_same(state, s):
                return idx
        return -1

    def update_state(self, bitmap):
        self.current_state = bitmap
        self.prev_time = self.current_time
        self.current_time = time.time()

    def get_first_guidance(self) -> Guidance:
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
        return Guidance(
            success=True,
            instruction=instruction,
            image=img_guidance,
            step_id=0
        )

    def get_guidance(self) -> Guidance:
        target_state = self.states[self.target_state_idx]

        # Check if we at least reached the previously desired state
        if bm.bitmap_same(self.current_state, target_state):

            if self.target_state_idx == len(self.states) - 1:
                # Task is done
                instruction = "You have completed the task. " \
                              "Congratulations!"
                # result['animation'] = bm.bitmap2guidance_animation(
                #     self.current_state, config.ACTION_TARGET)
                img_guidance = bm.bitmap2guidance_img(self.current_state, None,
                                                      config.ACTION_TARGET)
                return Guidance(
                    success=True,
                    instruction=instruction,
                    image=img_guidance,
                    step_id=self.target_state_idx
                )

            # Not done
            # Next state is simply the next one in line
            self.target_state_idx += 1
            target_state = self.states[self.target_state_idx]

            # Determine the type of change needed for the next step

            diff = bm.bitmap_diff(self.current_state, target_state)
            assert diff  # states can't be the same
            assert diff['n_diff_pieces'] == 1  # for now only change one
            # piece at the time

            self.prev_good_state = self.current_state
            self.good_word_idx = (self.good_word_idx +
                                  random.randint(1, 3)) % 4

            if diff['larger'] == 2:  # target state has one more piece
                instruction = bm.generate_message(
                    self.current_state,
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
                    self.current_state,
                    target_state,
                    config.ACTION_REMOVE,
                    diff['first_piece'],
                    step_time=self.current_time - self.prev_time,
                    good_word_idx=self.good_word_idx)

                # result['animation'] = bm.bitmap2guidance_animation(
                #    self.current_state,
                #     config.ACTION_REMOVE,
                #    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(self.current_state,
                                                      diff['first_piece'],
                                                      config.ACTION_REMOVE)

            return Guidance(
                success=True,
                instruction=instruction,
                image=img_guidance,
                step_id=self.target_state_idx
            )

        else:
            # reached an erroneous state
            self.target_state_idx = self.state2idx(self.prev_good_state)
            instruction = "This is incorrect, please undo the last " \
                          "step and revert to the model shown on " \
                          "the screen."
            # result['animation'] = bm.bitmap2guidance_animation(
            #     self.prev_good_state, config.ACTION_TARGET)
            img_guidance = bm.bitmap2guidance_img(self.prev_good_state,
                                                  None,
                                                  config.ACTION_TARGET)
            return Guidance(
                success=False,
                instruction=instruction,
                image=img_guidance,
                step_id=-1
            )
