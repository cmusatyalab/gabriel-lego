from __future__ import annotations

import random
import unittest
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from gabriel_lego.cv.colors import LEGOColorID


class _NotEnoughBricks(Exception):
    pass


class _NotEnoughSpace(Exception):
    pass


@dataclass
class Brick:
    length: int
    color: LEGOColorID

    def to_array_repr(self) -> List[int]:
        return [self.color.value] * self.length

    def __len__(self):
        return self.length

    def __getitem__(self, item: int):
        if type(item) != int:
            raise ValueError(item)
        elif item < 0 or item >= self.length:
            raise IndexError(item)
        else:
            return self.color.value

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.to_array_repr()}'

    def __hash__(self):
        return hash(str(self.length) + self.color.name)

    def __eq__(self, other: Brick):
        return self.color == other.color and self.length == other.length

    def copy(self) -> Brick:
        return Brick(self.length, self.color)


class BrickRow:
    def __init__(self, length: int = 6):
        self._length = length
        self._bricks = {}  # anchor: brick
        self._avail_positions = set(range(length))
        self._max_avail_space = length

    def clear(self):
        self._bricks = {}  # anchor: brick
        self._avail_positions = set(range(self._length))
        self._max_avail_space = self._length

    def copy(self) -> BrickRow:
        other = BrickRow()
        other._length = self._length
        for anchor, brick in self._bricks.items():
            other._bricks[anchor] = brick.copy()
        other._avail_positions = set()
        for pos in self._avail_positions:
            other._avail_positions.add(pos)

        other._max_avail_space = self._max_avail_space
        return other

    def __eq__(self, other: BrickRow):
        try:
            assert self._length == other._length
            assert len(self._bricks) == len(other._bricks)
            for anchor, brick in self._bricks.items():
                assert anchor in other._bricks
                assert other._bricks[anchor] == brick

            for pos in self._avail_positions:
                assert pos in other._avail_positions

            assert self._max_avail_space == other._max_avail_space
            return True
        except AssertionError:
            return False

    @property
    def brick_count(self) -> int:
        return len(self._bricks.keys())

    @property
    def length(self) -> int:
        return self._length

    @property
    def full(self) -> bool:
        return self._max_avail_space == 0

    @property
    def empty(self) -> bool:
        return self._max_avail_space == self._length

    @property
    def available_continuous_space(self) -> int:
        return self._max_avail_space

    def _update_avail_space(self) -> None:
        self._max_avail_space = 0
        for anchor in self._avail_positions:
            space = 0
            for i in range(anchor, self._length):
                if i in self._avail_positions:
                    space += 1
                else:
                    break

            self._max_avail_space = max(space, self._max_avail_space)

    def to_array_repr(self) -> List[int]:
        l_repr = [0] * self._length
        for anchor, brick in self._bricks.items():
            for i in range(anchor, anchor + brick.length):
                l_repr[i] = brick.color.value

        return l_repr

    def remove_random_brick(self) -> Brick:
        anchor, brick = random.choice(list(self._bricks.items()))
        del self._bricks[anchor]

        # update available space
        for i in range(anchor, anchor + brick.length):
            self._avail_positions.add(i)

        self._update_avail_space()
        return brick

    def add_brick(self, brick: Brick) -> None:
        if brick.length > self._max_avail_space:
            raise _NotEnoughSpace()

        fitting_anchors = []
        for anchor in self._avail_positions:
            # first peg of brick goes on top of anchor
            endpoint = anchor - 1 + brick.length
            if endpoint >= self._length:
                # tail of brick ends up outside of row
                continue
            else:
                fits = True
                for i in range(anchor, endpoint + 1):
                    if i not in self._avail_positions:
                        fits = False
                        break
                if fits:
                    fitting_anchors.append(anchor)

        anchor = random.choice(fitting_anchors)
        self._bricks[anchor] = brick

        # update available space
        for i in range(anchor, anchor + brick.length):
            self._avail_positions.remove(i)

        self._update_avail_space()


class BrickBoard:
    def __init__(self, base_brick: Brick):
        self._rows = []
        self._base = BrickRow(base_brick.length)
        self._width = base_brick.length

        self._base.add_brick(base_brick)
        self._base_brick = base_brick

    def copy(self) -> BrickBoard:
        other = BrickBoard(self._base_brick.copy())
        other._rows = [row.copy() for row in self._rows]

        return other

    def __eq__(self, other: BrickBoard):
        try:
            assert self.row_count == other.row_count
            assert self._base == other._base
            for row1, row2 in zip(self.rows(), other.rows()):
                assert row1 == row2
            return True
        except AssertionError:
            return False

    def rows(self) -> Iterator[BrickRow]:
        for row in self._rows:
            yield row

    @property
    def brick_count(self) -> int:
        return sum([row.brick_count for row in (self._rows + [self._base])])

    @property
    def base_color(self) -> LEGOColorID:
        return self._base_brick.color

    @property
    def row_count(self) -> int:
        return len(self._rows) + 1

    @property
    def empty(self) -> bool:
        return len(self._rows) == 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def avail_space_in_row(self):
        if len(self._rows) < 1 or self._rows[-1].full:
            return self._width
        else:
            return self._rows[-1].available_continuous_space

    def add_brick(self, brick: Brick) -> None:
        if len(self._rows) < 1 or self._rows[-1].full:
            self._rows.append(BrickRow(self._width))

        self._rows[-1].add_brick(brick)

    def remove_random_brick(self) -> Brick:
        if len(self._rows) < 1:
            raise _NotEnoughBricks()

        brick = self._rows[-1].remove_random_brick()

        if self._rows[-1].empty:
            self._rows.pop(-1)
        return brick

    def to_array_repr(self) -> List[List[int]]:
        return [row.to_array_repr() for row
                in reversed([self._base] + self._rows)]

    def clear(self):
        self._rows = []


class BrickCollection(object):
    def __init__(self, collection_dict=Dict[Brick, int]):
        super(BrickCollection, self).__init__()

        self._orig_collection = collection_dict.copy()

        self._collection = []
        for brick, count in collection_dict.items():
            self._collection += ([brick] * count)

    @property
    def brick_count(self) -> int:
        return len(self._collection)

    def reset(self):
        self._collection = []
        for brick, count in self._orig_collection.items():
            self._collection += ([brick] * count)

    def put_brick(self, brick: Brick) -> None:
        self._collection.append(brick)

    def get_brick(self, length: int, color: LEGOColorID) -> Optional[Brick]:
        tmp_brick = Brick(length, color)
        if tmp_brick in self._collection:
            self._collection.remove(tmp_brick)
            return tmp_brick
        else:
            return None

    def get_random_brick(self, max_len: int = 6) -> Brick:
        try:
            selected_brick = random.choice([b for b in self._collection
                                            if len(b) <= max_len])
            self._collection.remove(selected_brick)
        except IndexError as e:
            raise _NotEnoughBricks() from e

        return selected_brick


def gen_random_task(task_length: int,
                    collection: BrickCollection,
                    starting_board: BrickBoard) \
        -> List[BrickBoard]:
    assert task_length % 2 == 0

    states = [starting_board.copy()]

    step_cnt = 0
    while True:
        if starting_board.brick_count - 1 == task_length - step_cnt:
            break

        build = random.choice([True] * 5
                              + [False]) and collection.brick_count > 0

        if build or starting_board.empty:
            try:
                brick = collection.get_random_brick(
                    max_len=starting_board.avail_space_in_row)
                starting_board.add_brick(brick)
            except _NotEnoughBricks:
                collection.put_brick(starting_board.remove_random_brick())
        else:
            collection.put_brick(starting_board.remove_random_brick())

        states.append(starting_board.copy())
        step_cnt += 1

    while not starting_board.empty:
        collection.put_brick(starting_board.remove_random_brick())
        states.append(starting_board.copy())

    return states


def gen_random_latinsqr_task(min_task_len: int,
                             delays: List[float],
                             square_size: int,
                             collection: BrickCollection) -> List[pd.DataFrame]:
    avg_task_len = 2 * min_task_len
    max_task_len = 3 * min_task_len

    base_board = BrickBoard(collection.get_brick(6, LEGOColorID.RED))

    # min_task = gen_random_task(min_task_len, collection, base_board)
    # avg_task = gen_random_task(avg_task_len, collection, base_board)
    # max_task = gen_random_task(max_task_len, collection, base_board)

    combinations = []
    for d in delays:
        for task_len in (min_task_len, avg_task_len, max_task_len):
            # if random.choice([True, False]):
            #     task = list(reversed(task))
            combinations.append(
                (d, gen_random_task(task_len, collection, base_board)))

    sqr = [random.sample(combinations, k=len(combinations))]
    for i in range(1, square_size):
        sqr.append(sqr[i - 1][-1:] + sqr[i - 1][:-1])

    for i in range(square_size):
        sqr.append(list(reversed(sqr[i])))

    # unroll everything
    latin_sqr = []
    for i in range(len(sqr)):
        task_seq = sqr[i]
        task_df = pd.DataFrame(columns=['delay', 'state'])

        task_df = task_df.append({
            'delay': 0,
            'state': base_board.to_array_repr()
        }, ignore_index=True)

        for d, steps in task_seq:
            for step in steps[1:]:  # skip first step in each subtask
                task_df = task_df.append({
                    'delay': d,
                    'state': step.to_array_repr()
                }, ignore_index=True)
        latin_sqr.append(task_df)

    return latin_sqr


Life_of_George_Bricks = BrickCollection(
    collection_dict={
        # black bricks
        # Brick(1, LEGOColorID.BLACK) : 8,
        # Brick(2, LEGOColorID.BLACK) : 6,
        # Brick(6, LEGOColorID.BLACK) : 2,
        # Brick(4, LEGOColorID.BLACK) : 4,
        # Brick(3, LEGOColorID.BLACK) : 4,
        # blue bricks
        Brick(1, LEGOColorID.BLUE)  : 6,
        Brick(2, LEGOColorID.BLUE)  : 8,
        Brick(6, LEGOColorID.BLUE)  : 2,
        Brick(4, LEGOColorID.BLUE)  : 4,
        Brick(3, LEGOColorID.BLUE)  : 4,
        # red bricks
        Brick(1, LEGOColorID.RED)   : 6,
        Brick(2, LEGOColorID.RED)   : 8,
        Brick(6, LEGOColorID.RED)   : 2,
        Brick(4, LEGOColorID.RED)   : 4,
        Brick(3, LEGOColorID.RED)   : 4,
        # yellow bricks
        Brick(1, LEGOColorID.YELLOW): 6,
        Brick(2, LEGOColorID.YELLOW): 8,
        Brick(6, LEGOColorID.YELLOW): 2,
        Brick(4, LEGOColorID.YELLOW): 4,
        Brick(3, LEGOColorID.YELLOW): 4,
        # green bricks
        Brick(1, LEGOColorID.GREEN) : 6,
        Brick(2, LEGOColorID.GREEN) : 8,
        Brick(6, LEGOColorID.GREEN) : 2,
        Brick(4, LEGOColorID.GREEN) : 4,
        Brick(3, LEGOColorID.GREEN) : 4,
        # white bricks
        # Brick(1, LEGOColorID.WHITE) : 8,
        # Brick(2, LEGOColorID.WHITE) : 6,
        # Brick(6, LEGOColorID.WHITE) : 2,
        # Brick(4, LEGOColorID.WHITE) : 4,
        # Brick(3, LEGOColorID.WHITE) : 4,
    }
)


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._valid_colors = [color for color in LEGOColorID
                              if color != LEGOColorID.NOTHING]

        self._row = BrickRow(6)
        self._board = BrickBoard(
            Life_of_George_Bricks.get_brick(6, LEGOColorID.RED))

    def tearDown(self) -> None:
        self._row.clear()
        self._board.clear()
        Life_of_George_Bricks.reset()

    def test_bricks(self):
        for color_id in self._valid_colors:
            length = random.randint(1, 10)
            brick = Brick(length, color_id)

            self.assertListEqual(brick.to_array_repr(),
                                 [color_id.value] * length)

    def test_add_brick_to_empty_row(self):
        for i in range(1, 7):
            brick_cnt = self._row.brick_count
            brick = Brick(i, random.choice(self._valid_colors))
            self._row.add_brick(brick)

            self.assertLess(self._row.available_continuous_space,
                            self._row.length)
            self.assertEqual(self._row.brick_count, brick_cnt + 1)

            if brick.length == self._row.length:
                self.assertEqual(0, self._row.available_continuous_space)

            self._row.clear()

    def test_fill_row(self):
        self._row.add_brick(Brick(int(np.floor(self._row.length / 2)),
                                  random.choice(self._valid_colors)))

        while not self._row.full:
            self._row.add_brick(Brick(self._row._max_avail_space,
                                      random.choice(self._valid_colors)))

        self.assertNotIn(0, self._row.to_array_repr(),
                         self._row.to_array_repr())

    def test_remove_brick(self):
        # fill row first
        self.test_fill_row()

        brick_cnt = self._row.brick_count
        brick = self._row.remove_random_brick()
        self.assertEqual(brick_cnt - 1, self._row.brick_count)
        self.assertEqual(brick.length, self._row.available_continuous_space,
                         (self._row.to_array_repr(), brick.to_array_repr()))

    def test_empty_row(self):
        # fill row first
        self.test_fill_row()
        while not self._row.empty:
            _ = self._row.remove_random_brick()

        self.assertEqual(self._row.to_array_repr(),
                         [0] * self._row.length)
        self.assertEqual(0, self._row.brick_count)

    def test_add_too_big_brick(self):
        # fill row first
        self.test_fill_row()

        brick = self._row.remove_random_brick()

        brick_cnt = self._row.brick_count
        with self.assertRaises(_NotEnoughSpace):
            self._row.add_brick(Brick(brick.length + 1, brick.color))
        self.assertEqual(brick_cnt, self._row.brick_count)

    def test_replace_brick_with_smaller_bricks(self):
        # fill row first
        self.test_fill_row()

        brick_cnt = self._row.brick_count

        brick = self._row.remove_random_brick()
        while brick.length <= 1:
            self._row.add_brick(brick)
            brick = self._row.remove_random_brick()

        # split brick
        brick_1 = Brick(brick.length - 1, brick.color)
        brick_2 = Brick(1, brick.color)

        self._row.add_brick(brick_1)
        self._row.add_brick(brick_2)

        self.assertTrue(self._row.full)
        self.assertNotIn(0, self._row.to_array_repr())

        self.assertEqual(brick_cnt + 1, self._row.brick_count)

    def test_init_table(self):
        self.assertEqual(1, self._board.row_count)
        l_repr = self._board.to_array_repr()
        self.assertListEqual(l_repr,
                             [[self._board.base_color.value]
                              * self._board.width])
        self.assertEqual(1, self._board.brick_count)

    def test_add_brick_empty_table(self):
        self.assertEqual(1, self._board.row_count)
        brick = Brick(random.randint(1, self._board.width),
                      random.choice(self._valid_colors))

        self._board.add_brick(brick)
        self.assertEqual(2, self._board.row_count)
        self.assertEqual(2, self._board.brick_count)

    def test_fill_table(self):
        self.assertEqual(1, self._board.row_count)

        brick_cnt = self._board.brick_count

        # add a 100 rows
        for i in range(100):
            self._board.add_brick(Brick(int(np.floor(self._board.width / 2)),
                                        random.choice(self._valid_colors)))

            brick_cnt += 1
            self.assertEqual(brick_cnt, self._board.brick_count)

            while self._board.avail_space_in_row < self._board.width:
                self._board.add_brick(Brick(self._board.avail_space_in_row,
                                            random.choice(self._valid_colors)))

                brick_cnt += 1
                self.assertEqual(brick_cnt, self._board.brick_count)

        self.assertEqual(100 + 1, self._board.row_count)

        for row in self._board.rows():
            self.assertTrue(row.full)
            self.assertNotIn(0, row.to_array_repr())

    def test_empty_table(self):
        # fill first
        self.test_fill_table()

        brick_cnt = self._board.brick_count
        while not self._board.empty:
            _ = self._board.remove_random_brick()
            brick_cnt -= 1
            self.assertEqual(brick_cnt, self._board.brick_count)

        self.assertEqual(1, self._board.brick_count)
        self.assertEqual(1, self._board.row_count)

    def test_random_task_gen(self):
        steps = 200
        task = gen_random_task(steps, Life_of_George_Bricks, self._board)

        self.assertEqual(self._board, task[0])
        self.assertEqual(self._board, task[-1])

        for i in range(1, len(task)):
            state1 = task[i - 1]
            state2 = task[i]
            self.assertEqual(1, abs(state1.brick_count - state2.brick_count))

        self.assertEqual(steps + 1, len(task))


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    tasks = gen_random_latinsqr_task(
        min_task_len=6,
        delays=[0, .600, 1.200, 1.800, 2.400, 3.000],
        square_size=6,
        collection=Life_of_George_Bricks
    )

    for i, t in enumerate(tasks):
        print(t, end='\n------------------------------------------------\n')
        t.to_csv(f'./latin_sqr_{i}.csv', index=False)

    # gen test task
    s_board = BrickBoard(Life_of_George_Bricks.get_brick(6, LEGOColorID.RED))
    test_task = gen_random_task(6, Life_of_George_Bricks, s_board)

    test_task_df = pd.DataFrame(columns=['delay', 'state'])
    for state in test_task:
        test_task_df = test_task_df.append({
            'delay': 0,
            'state': state.to_array_repr()
        }, ignore_index=True)

    print(test_task_df)
    test_task_df.to_csv(f'./test_task.csv', index=False)
