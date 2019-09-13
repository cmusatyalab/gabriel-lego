from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from gabriel_lego.cv import bitmap as bm
from gabriel_lego.lego_engine import config, tasks
from gabriel_lego.lego_engine.board import BoardState, EmptyBoardState


class NoStateChangeError(Exception):
    pass


class NoGuidanceError(Exception):
    pass


class NoSuchTaskError(Exception):
    pass


class UnreachableStateError(Exception):
    pass


def get_guidance_between_states(from_state: BoardState,
                                to_state: BoardState) -> Tuple[str, np.ndarray]:
    if from_state == to_state:
        raise NoStateChangeError()

    diff = from_state.diff(to_state)
    assert diff is not None
    if diff['n_diff_pieces'] > 1:
        raise UnreachableStateError()

    good_word_idx = random.randint(0, 100) % 4

    if diff['larger'] == 2:  # target state has one more piece
        instruction = bm.generate_message(
            from_state.bitmap,
            to_state.bitmap,
            config.ACTION_ADD,
            diff['first_piece'],
            good_word_idx=good_word_idx)

        img_guidance = bm.bitmap2guidance_img(to_state.bitmap,
                                              diff['first_piece'],
                                              config.ACTION_ADD)
    else:  # target state has one less piece
        instruction = bm.generate_message(
            from_state.bitmap,
            to_state.bitmap,
            config.ACTION_REMOVE,
            diff['first_piece'],
            good_word_idx=good_word_idx)

        img_guidance = bm.bitmap2guidance_img(from_state.bitmap,
                                              diff['first_piece'],
                                              config.ACTION_REMOVE)

    return instruction, img_guidance


def get_error_guidance(target_state: BoardState) \
        -> Tuple[str, Optional[np.ndarray]]:
    if target_state.empty_board:
        instruction = 'Incorrect. Please clear the LEGO board to continue.'
        return instruction, None

    instruction = "This is incorrect, please revert to the model shown on " \
                  "the screen."

    img_guidance = bm.bitmap2guidance_img(target_state.bitmap,
                                          None,
                                          config.ACTION_TARGET)

    return instruction, img_guidance


class TaskState(ABC):
    @staticmethod
    def generate_correct_state(task: List[BoardState], state_index: int) \
            -> Union[CorrectTaskState, FinalTaskState]:
        assert state_index in range(len(task))

        if state_index == len(task) - 1:
            return FinalTaskState(task)
        else:
            return CorrectTaskState(task, state_index)

    def __init__(self):
        self.current_image = None
        self.current_instruction = None

    @abstractmethod
    def compute_next_task_state(self, new_board_state: BoardState) -> TaskState:
        pass

    def get_guidance(self):
        return self.current_instruction, self.current_image

    @abstractmethod
    def get_current_board_state(self) -> BoardState:
        pass


class InitialTaskState(TaskState):
    def __init__(self, task: List[BoardState]):
        super().__init__()
        self.task = task
        self.current_instruction = 'To start, put the empty LEGO board into ' \
                                   'focus.'

    def compute_next_task_state(self, new_board_state: BoardState) -> TaskState:
        if not new_board_state.empty_board:
            raise NoStateChangeError()

        return CorrectTaskState(self.task, 0)

    def get_current_board_state(self) -> BoardState:
        raise RuntimeError('Initial task state does not have a board state.')


class CorrectTaskState(TaskState):
    def __init__(self,
                 task: List[BoardState],
                 current_state_index: int):
        super().__init__()
        assert current_state_index in range(len(task) - 1)

        self.task = task

        self.current_state = task[current_state_index]
        target_state = task[current_state_index + 1]

        instruction, image = get_guidance_between_states(self.current_state,
                                                         target_state)

        self.current_state_index = current_state_index
        self.current_instruction = instruction
        self.current_image = image

    def compute_next_task_state(self, new_board_state: BoardState) -> TaskState:
        if self.current_state == new_board_state:
            raise NoStateChangeError()

        target_state_index = self.current_state_index + 1
        target_state = self.task[target_state_index]
        # Check if we reached the next state
        if target_state == new_board_state:
            # Next state is simply the next one in line
            return TaskState.generate_correct_state(self.task,
                                                    target_state_index)

        else:
            return IncorrectTaskState(
                task=self.task,
                target_state_index=self.current_state_index,
                current_state=new_board_state)

    def get_current_board_state(self) -> BoardState:
        return self.current_state


class FinalTaskState(TaskState):
    def __init__(self, task: List[BoardState]):
        super().__init__()
        self.task = task
        self.current_instruction = "You have completed the task. " \
                                   "Congratulations!"
        self.current_image = bm.bitmap2guidance_img(task[-1].bitmap, None,
                                                    config.ACTION_TARGET)

    def compute_next_task_state(self, new_board_state: BoardState) -> TaskState:
        raise NoStateChangeError()

    def get_current_board_state(self) -> BoardState:
        return self.task[-1]


class IncorrectTaskState(TaskState):
    def __init__(self,
                 task: List[BoardState],
                 target_state_index: int,
                 current_state: BoardState):
        super().__init__()
        assert target_state_index in range(len(task))

        self.target_state_index = target_state_index
        self.target_state = task[target_state_index]

        instruction, image = get_error_guidance(self.target_state)

        self.task = task
        self.current_state = current_state
        self.current_instruction = instruction
        self.current_image = image

    def compute_next_task_state(self, new_board_state: BoardState) -> TaskState:
        if new_board_state == self.current_state:
            raise NoStateChangeError()

        if new_board_state == self.target_state:
            # fixed error
            return CorrectTaskState(
                task=self.task,
                current_state_index=self.target_state_index)

        else:
            return IncorrectTaskState(
                task=self.task,
                target_state_index=self.target_state_index,
                current_state=new_board_state)

    def get_current_board_state(self) -> BoardState:
        return self.current_state


class TaskManager:
    def __init__(self):
        self.tasks = {
            name: [EmptyBoardState()] + [BoardState(b) for b in bitmaps]
            for name, bitmaps in tasks.task_collection.items()
        }

    def _get_task(self, task_name: str) -> List[BoardState]:
        try:
            return self.tasks[task_name]
        except KeyError:
            raise NoSuchTaskError(task_name)

        # todo: maybe return list of valid task names?
