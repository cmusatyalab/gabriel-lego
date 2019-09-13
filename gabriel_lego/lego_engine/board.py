from __future__ import annotations

from typing import Dict

import numpy as np

from gabriel_lego.cv import bitmap as bm


class BoardState:
    def __init__(self, bitmap: np.ndarray):
        self.bitmap = bitmap
        self.empty_board = False

    def __eq__(self, other: BoardState) -> bool:
        return bm.bitmap_same(self.bitmap, other.bitmap)

    def diff(self, other: BoardState) -> Dict:
        return bm.bitmap_diff(self.bitmap, other.bitmap)


class EmptyBoardState(BoardState):
    def __init__(self):
        super().__init__(np.array([[]], dtype='int32'))
        self.empty_board = True
