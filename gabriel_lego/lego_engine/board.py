from __future__ import annotations

from typing import Dict

import numpy as np

from gabriel_lego.cv import bitmap as bm


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
