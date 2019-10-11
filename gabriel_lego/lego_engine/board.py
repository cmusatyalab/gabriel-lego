from __future__ import annotations

from typing import Dict

import numpy as np

from gabriel_lego.cv import bitmap as bm


class BoardState:
    def __init__(self, bitmap: np.ndarray):
        self._bitmap = bitmap
        self._empty_board = False

    @property
    def bitmap(self):
        return self._bitmap

    @property
    def empty_board(self):
        return self._empty_board

    def __eq__(self, other: BoardState) -> bool:
        return bm.bitmap_same(self._bitmap, other._bitmap)

    def diff(self, other: BoardState) -> Dict:
        return bm.bitmap_diff(self._bitmap, other._bitmap)


class EmptyBoardState(BoardState):
    def __init__(self):
        super().__init__(np.array([[]], dtype='int32'))
        self._empty_board = True
