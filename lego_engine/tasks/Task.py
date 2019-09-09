from __future__ import annotations

import random
import time
from typing import Dict, NamedTuple, Optional

import numpy as np

from cv import bitmap as bm
from lego_engine import config


class Guidance(NamedTuple):
    success: bool
    instruction: str
    image: Optional[np.ndarray]
    step_id: int


class BoardState:
    def __init__(self, bitmap: np.ndarray):
        self.bitmap = bitmap

    def __eq__(self, other: BoardState) -> bool:
        return bm.bitmap_same(self.bitmap, other.bitmap)

    def empty_board(self) -> bool:
        return False

    def diff(self, other: BoardState) -> Dict:
        return bm.bitmap_diff(self.bitmap, other.bitmap)


class EmptyBoardState(BoardState):
    def __init__(self):
        super().__init__(np.zeros(1))

    def __eq__(self, other: BoardState) -> bool:
        return type(other) == EmptyBoardState

    def empty_board(self) -> bool:
        return True

    def diff(self, other: BoardState) -> Dict:
        raise RuntimeError('Cannot compare empty board to other states.')


class NullBoardState(BoardState):
    def __init__(self):
        super().__init__(np.zeros(1))

    def __eq__(self, other: BoardState) -> bool:
        return False

    def empty_board(self) -> bool:
        raise RuntimeError('Null state error.')

    def diff(self, other: BoardState) -> Dict:
        raise RuntimeError('Cannot compare Null state to other states.')


class NoStateChangeError(Exception):
    pass


class NoGuidanceError(Exception):
    pass


class Task:
    def __init__(self, bitmaps):
        self.states = [BoardState(bitmap) for bitmap in bitmaps]
        self.current_state = NullBoardState()

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
            "board." % (target_state.bitmap.shape[1],
                        config.COLOR_ORDER[target_state.bitmap[0, 0]])
        # animation = bm.bitmap2guidance_animation(target_state,
        #                                         config.ACTION_TARGET)
        img_guidance = bm.bitmap2guidance_img(target_state.bitmap, None,
                                              config.ACTION_TARGET)
        self.initial_guidance = Guidance(
            success=True,
            instruction=instruction,
            image=img_guidance,
            step_id=0
        )

        self.current_guidance = None

    def update_state(self, state: BoardState) -> None:
        if state == self.current_state:
            # raise exception if new state is same as old state
            # using exceptions for flow control is Pythonic, don't @ me though
            raise NoStateChangeError()

        self.current_state = state

        self.prev_time = self.current_time
        self.current_time = time.time()

        # if incoming new state is an empty board and target state is the
        # first step in the task, set initial guidance
        if state.empty_board() and self.target_state_idx == 0:
            self.current_guidance = self.initial_guidance
            return

        target_state = self.states[self.target_state_idx]

        # Check if we at least reached the previously desired state
        # if bm.bitmap_same(new_state, target_state):
        if target_state == state:
            if self.target_state_idx == (len(self.states) - 1):
                # Task is done
                instruction = "You have completed the task. " \
                              "Congratulations!"
                # result['animation'] = bm.bitmap2guidance_animation(
                #     self.current_state, config.ACTION_TARGET)
                img_guidance = bm.bitmap2guidance_img(state.bitmap, None,
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

            # diff = bm.bitmap_diff(new_state, target_state)
            diff = state.diff(target_state)
            assert diff  # states can't be the same
            assert diff['n_diff_pieces'] == 1  # for now only change one
            # piece at the time

            self.good_word_idx = (self.good_word_idx +
                                  random.randint(1, 3)) % 4

            if diff['larger'] == 2:  # target state has one more piece
                instruction = bm.generate_message(
                    state.bitmap,
                    target_state.bitmap,
                    config.ACTION_ADD,
                    diff['first_piece'],
                    step_time=self.current_time - self.prev_time,
                    good_word_idx=self.good_word_idx)

                # result['animation'] = bm.bitmap2guidance_animation(
                #    target_state,
                #    config.ACTION_ADD,
                #    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(target_state.bitmap,
                                                      diff['first_piece'],
                                                      config.ACTION_ADD)
            else:  # target state has one less piece
                instruction = bm.generate_message(
                    state.bitmap,
                    target_state.bitmap,
                    config.ACTION_REMOVE,
                    diff['first_piece'],
                    step_time=self.current_time - self.prev_time,
                    good_word_idx=self.good_word_idx)

                # result['animation'] = bm.bitmap2guidance_animation(
                #    self.current_state,
                #     config.ACTION_REMOVE,
                #    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(state.bitmap,
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
            img_guidance = bm.bitmap2guidance_img(target_state.bitmap,
                                                  None,
                                                  config.ACTION_TARGET)
            self.current_guidance = Guidance(
                success=False,
                instruction=instruction,
                image=img_guidance,
                step_id=-1
            )

    def get_guidance(self) -> Guidance:
        if self.current_guidance is None:
            raise NoGuidanceError()
        return self.current_guidance
