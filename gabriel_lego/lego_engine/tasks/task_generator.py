from __future__ import annotations

import queue
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from gabriel_lego.cv.colors import LEGOColorID


class NotEnoughBricks(Exception):
    pass


@dataclass
class Brick:
    length: int
    color: LEGOColorID

    def to_array_rep(self) -> List[int]:
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
        return f'{self.to_array_rep()}'

    def __hash__(self):
        return hash(str(self.length) + self.color.name)

    def __eq__(self, other: Brick):
        return self.color == other.color and self.length == other.length


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
            raise NotEnoughBricks() from e

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
                except NotEnoughBricks:
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

if __name__ == '__main__':
    import pprint
    import io

    # 270 is the approx num of steps necessary for a 25 minute-long task
    for num_steps in [20, 45, 90, 135, 180, 270]:
        task = DefaultGenerator.generate(num_steps, height=7)
        t_string = io.StringIO()
        pprint.pprint(task, stream=t_string)

        with open(f'./task_generated_{num_steps}.py', 'w') as fp:
            print(
                f'''
from numpy import array

# Automatically generated task with {num_steps} steps

# Labels: nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6,
# unsure:7
bitmaps = \\
{t_string.getvalue()}
''',
                file=fp
            )
