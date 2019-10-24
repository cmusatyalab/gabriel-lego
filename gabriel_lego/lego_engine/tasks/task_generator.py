from __future__ import annotations

import queue
import random
import unittest
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

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


class BrickTable:
    def __init__(self, width: int = 6,
                 base_color: LEGOColorID = LEGOColorID.RED):
        self._rows = []
        self._base = BrickRow(width)
        self._width = width

        self._base.add_brick(Brick(width, base_color))
        self._base_color = base_color

    def rows(self) -> Iterator[BrickRow]:
        for row in self._rows:
            yield row

    @property
    def brick_count(self) -> int:
        return sum([row.brick_count for row in (self._rows + [self._base])])

    @property
    def base_color(self) -> LEGOColorID:
        return self._base_color

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


class TaskGenerator(object):
    def __init__(self, collection):
        super(TaskGenerator, self).__init__()
        self.collection = collection

    @staticmethod
    def check_anchor(anchor, brick, table):
        height, width = table.shape
        # first check: does brick fit in table?
        if anchor + len(brick) - 1 >= width:
            return False

        ret = False
        for i in range(anchor, anchor + len(brick)):
            # second check, does it clash with any other brick
            if table[0][i] != 0:
                return False

            # third check: are there any support points
            if 1 == height:
                ret = True
            elif table[1][i] != 0:
                ret = True

        return ret

    @staticmethod
    def add_brick(table, brick):
        n_table = np.copy(table)
        width = table.shape[1]

        anchor = random.choice([i for i in range(width)
                                if
                                TaskGenerator.check_anchor(i, brick, n_table)])

        for i in range(anchor, anchor + len(brick)):
            n_table[0][i] = brick[i - anchor]

        return n_table

    @staticmethod
    def find_max_space_rem(table):
        current_max = 0
        tmp_cnt = 0
        i = 0
        while i <= table.shape[1]:
            if i == 0:
                tmp_cnt += 1
            else:
                tmp_cnt = 0

            current_max = max(tmp_cnt, current_max)
            i += 1

        return current_max

    def generate(self, num_steps, height=4, base_brick_color=LEGOColorID.RED):
        assert num_steps >= 1

        base = self.collection.get_brick(length=6, color=base_brick_color)

        steps = []
        base_table = np.full((1, len(base)),
                             fill_value=base_brick_color.value,
                             dtype=int)
        steps.append(base_table)
        table = np.vstack((np.zeros((1, len(base)), dtype=int), base_table))

        current_level = 1
        adding = True
        temp_stack = queue.LifoQueue()
        max_rem_space = len(base)
        while len(steps) < num_steps:
            if adding:
                try:
                    brick = self.collection.get_random_brick(
                        max_len=max_rem_space)
                    table = self.add_brick(table, brick)
                    max_rem_space = TaskGenerator.find_max_space_rem(table)
                    n_table = np.copy(table)
                    steps.append(n_table)
                    temp_stack.put_nowait((n_table, brick))
                except IndexError:
                    # level is full
                    current_level += 1
                    max_rem_space = len(base)

                    # chance to switch directions will always be 100% at the
                    # final height since diff will be 0
                    chance_to_switch = ([True] * current_level) + \
                                       ([False] * (height - current_level))

                    if random.choice(chance_to_switch):
                        adding = False
                        temp_stack.get_nowait()  # pop the latest step
                    else:
                        table = np.vstack((np.zeros((1, len(base)),
                                                    dtype=int), table))
                except _NotEnoughBricks:
                    # not enough bricks, we HAVE to tear down to get them back
                    adding = False
                    temp_stack.get_nowait()  # pop the latest step
            else:
                try:
                    step, brick = temp_stack.get_nowait()
                    steps.append(step)
                    self.collection.put_brick(brick)
                except queue.Empty:
                    steps.append(base_table)
                    table = np.vstack(
                        (np.zeros((1, len(base)), dtype=int), base_table))
                    current_level = 1
                    adding = True
                    max_rem_space = len(base)

        self.collection.reset()
        return steps


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

DefaultGenerator = TaskGenerator(Life_of_George_Bricks)


class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._valid_colors = [color for color in LEGOColorID
                              if color != LEGOColorID.NOTHING]

        self._row = BrickRow(6)
        self._table = BrickTable(width=6,
                                 base_color=LEGOColorID.RED)

    def tearDown(self) -> None:
        self._row.clear()
        self._table.clear()

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
        self.assertEqual(1, self._table.row_count)
        l_repr = self._table.to_array_repr()
        self.assertListEqual(l_repr,
                             [[self._table.base_color.value]
                              * self._table.width])
        self.assertEqual(1, self._table.brick_count)

    def test_add_brick_empty_table(self):
        self.assertEqual(1, self._table.row_count)
        brick = Brick(random.randint(1, self._table.width),
                      random.choice(self._valid_colors))

        self._table.add_brick(brick)
        self.assertEqual(2, self._table.row_count)
        self.assertEqual(2, self._table.brick_count)

    def test_fill_table(self):
        self.assertEqual(1, self._table.row_count)

        brick_cnt = self._table.brick_count

        # add a 100 rows
        for i in range(100):
            self._table.add_brick(Brick(int(np.floor(self._table.width / 2)),
                                        random.choice(self._valid_colors)))

            brick_cnt += 1
            self.assertEqual(brick_cnt, self._table.brick_count)

            while self._table.avail_space_in_row < self._table.width:
                self._table.add_brick(Brick(self._table.avail_space_in_row,
                                            random.choice(self._valid_colors)))

                brick_cnt += 1
                self.assertEqual(brick_cnt, self._table.brick_count)

        self.assertEqual(100 + 1, self._table.row_count)

        for row in self._table.rows():
            self.assertTrue(row.full)
            self.assertNotIn(0, row.to_array_repr())

    def test_empty_table(self):
        # fill first
        self.test_fill_table()

        brick_cnt = self._table.brick_count
        while not self._table.empty:
            _ = self._table.remove_random_brick()
            brick_cnt -= 1
            self.assertEqual(brick_cnt, self._table.brick_count)

        self.assertEqual(1, self._table.brick_count)
        self.assertEqual(1, self._table.row_count)
