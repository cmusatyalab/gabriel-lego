from __future__ import annotations

import random
import time
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from gabriel_lego.cv import bitmap as bm
from gabriel_lego.lego_engine import config, tasks


class Guidance(NamedTuple):
    success: bool
    instruction: str
    image: Optional[np.ndarray]
    target_state_index: int
    previous_state_index: int
    task_finished: bool


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


class NoSuchTaskError(Exception):
    pass


class TaskManager:
    def __init__(self):
        self.tasks = {
            name: [BoardState(b) for b in bitmaps]
            for name, bitmaps in tasks.task_collection.items()
        }

    def _get_task(self, task_name: str) -> List[BoardState]:
        try:
            return self.tasks[task_name]
        except KeyError:
            raise NoSuchTaskError(task_name)

        # todo: maybe return list of valid task names?

    def get_guidance(self,
                     task_name: str,
                     board_state: BoardState,
                     target_state_index: int,
                     previous_state_index: int,
                     prev_timestamp: float) -> Guidance:

        task = self._get_task(task_name)
        target_state = task[target_state_index] \
            if target_state_index in range(len(task)) else NullBoardState()
        prev_state = task[previous_state_index] \
            if previous_state_index in range(len(task)) else NullBoardState()

        if target_state_index != previous_state_index and \
                board_state == prev_state:
            # raise exception if new state is same as old state
            # using exceptions for flow control is Pythonic, don't @ me though
            raise NoStateChangeError()

        # if just starting the task, make sure we're seeing the bare board
        # and send the initial guidance
        if target_state_index == -1:
            if board_state.empty_board():
                target_state = task[0]

                instruction = \
                    "Welcome to the Lego task. As a first step, please " \
                    "find a piece of 1x%d %s brick and put it on the " \
                    "board." % (target_state.bitmap.shape[1],
                                config.COLOR_ORDER[target_state.bitmap[0, 0]])
                # animation = bm.bitmap2guidance_animation(target_state,
                #                                         config.ACTION_TARGET)
                img_guidance = bm.bitmap2guidance_img(target_state.bitmap, None,
                                                      config.ACTION_TARGET)
                return Guidance(
                    success=True,
                    instruction=instruction,
                    image=img_guidance,
                    target_state_index=0,
                    previous_state_index=-1,
                    task_finished=False
                )
            else:
                return Guidance(
                    success=False,
                    instruction="To start, please clear the LEGO board.",
                    image=None,
                    target_state_index=-1,
                    previous_state_index=-1,
                    task_finished=False
                )

        # Check if we at least reached the previously desired state
        # if bm.bitmap_same(new_state, target_state):
        if target_state == board_state:
            if target_state_index == (len(task) - 1):
                # Task is done
                instruction = "You have completed the task. " \
                              "Congratulations!"
                # result['animation'] = bm.bitmap2guidance_animation(
                #     self.current_state, config.ACTION_TARGET)
                img_guidance = bm.bitmap2guidance_img(board_state.bitmap, None,
                                                      config.ACTION_TARGET)
                return Guidance(
                    success=True,
                    instruction=instruction,
                    image=img_guidance,
                    target_state_index=-1,
                    previous_state_index=-1,
                    task_finished=True
                )

            # Not done
            # Next state is simply the next one in line
            previous_state_index = target_state_index
            target_state_index = target_state_index + 1
            target_state = task[target_state_index]

            # Determine the type of change needed for the next step

            # diff = bm.bitmap_diff(new_state, target_state)
            diff = board_state.diff(target_state)
            assert diff  # states can't be the same
            assert diff['n_diff_pieces'] == 1  # for now only change one
            # piece at the time

            good_word_idx = random.randint(0, 100) % 4

            if diff['larger'] == 2:  # target state has one more piece
                instruction = bm.generate_message(
                    board_state.bitmap,
                    target_state.bitmap,
                    config.ACTION_ADD,
                    diff['first_piece'],
                    step_time=time.time() - prev_timestamp,
                    good_word_idx=good_word_idx)

                # result['animation'] = bm.bitmap2guidance_animation(
                #    target_state,
                #    config.ACTION_ADD,
                #    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(target_state.bitmap,
                                                      diff['first_piece'],
                                                      config.ACTION_ADD)
            else:  # target state has one less piece
                instruction = bm.generate_message(
                    board_state.bitmap,
                    target_state.bitmap,
                    config.ACTION_REMOVE,
                    diff['first_piece'],
                    step_time=time.time() - prev_timestamp,
                    good_word_idx=good_word_idx)

                # result['animation'] = bm.bitmap2guidance_animation(
                #    self.current_state,
                #     config.ACTION_REMOVE,
                #    diff_piece=diff['first_piece'])

                img_guidance = bm.bitmap2guidance_img(board_state.bitmap,
                                                      diff['first_piece'],
                                                      config.ACTION_REMOVE)

            return Guidance(
                success=True,
                instruction=instruction,
                image=img_guidance,
                target_state_index=target_state_index,
                previous_state_index=previous_state_index,
                task_finished=False
            )

        else:
            # reached an erroneous state
            if previous_state_index in range(len(task)):
                target_state_index = previous_state_index
                instruction = "This is incorrect, please undo the last " \
                              "step and revert to the model shown on " \
                              "the screen."
                # result['animation'] = bm.bitmap2guidance_animation(
                #     self.prev_good_state, config.ACTION_TARGET)
                target_state = task[target_state_index]
                img_guidance = bm.bitmap2guidance_img(target_state.bitmap,
                                                      None,
                                                      config.ACTION_TARGET)
                return Guidance(
                    success=False,
                    instruction=instruction,
                    image=img_guidance,
                    target_state_index=target_state_index,
                    previous_state_index=-1,
                    task_finished=False
                )

            else:

                return Guidance(
                    success=False,
                    instruction="This is incorrect. Please clear the LEGO "
                                "board to continue.",
                    image=None,
                    target_state_index=-1,
                    previous_state_index=-1,
                    task_finished=False
                )
