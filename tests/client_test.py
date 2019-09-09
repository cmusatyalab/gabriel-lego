import logging
import random
import sys
import unittest

from lego_engine.engine import LEGOEngine
from lego_engine.protocol import GabrielInput, GabrielOutput
from lego_engine.tasks import Task, task_Turtle


class ClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger('TestDebug')
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

        self.frames = []
        for frame_file in [f'./frames/step_{i}.jpeg' for i in range(8)]:
            with open(frame_file, 'rb') as f:
                # self.logger.debug(f'Loading {frame_file} into memory...')
                self.frames.append(f.read())

        with open('./cv_bad_frame.jpeg', 'rb') as f:
            self.junk_frame = f.read()

        self.engine = LEGOEngine(Task(task_Turtle.bitmaps))

    @staticmethod
    def prepare_protobuf(frame_id: int, image: bytes) -> GabrielInput:
        proto = GabrielInput()
        proto.type = GabrielInput.Type.IMAGE
        proto.frame_id = frame_id
        proto.payload = image
        return proto

    def test_perfect_run(self, yield_after_step=False):
        for i, frame in enumerate(self.frames):
            msg = self.prepare_protobuf(i, frame)
            res = self.engine.handle_request(msg)

            self.assertEqual(res.frame_id, i)
            self.assertEqual(res.status,
                             GabrielOutput.Status.SUCCESS,
                             msg=f'{res}')
            self.assertEqual(len(res.results), 2)
            if yield_after_step:
                yield i, msg

    def test_junk_frames(self):
        # execute "perfect" run with a bunch of "junk" frames between each
        # correct step, similar to how a real run would go

        for _, _ in self.test_perfect_run(yield_after_step=True):
            # insert [10, 20) junk frames between each step
            # the junk frames should not alter the execution of the task
            for i in range(random.randint(10, 20)):
                frame_id = random.randint(0, 500)
                msg = self.prepare_protobuf(frame_id, self.junk_frame)
                res = self.engine.handle_request(msg)

                self.assertEqual(res.frame_id, frame_id)
                self.assertEqual(res.status,
                                 GabrielOutput.Status.ENGINE_ERROR,
                                 msg=f'{res}')

    def test_mistake_recovery(self):
        # execute a run where a mistake is made and recovered after each step

        for i, msg in self.test_perfect_run(yield_after_step=True):
            # insert a human mistake after each step
            # a human mistake is exemplified by a valid frame for a wrong step
            wrong_frame_id = random.choice(
                [j for j in range(len(self.frames)) if j != i + 1])

            mistake_msg = self.prepare_protobuf(i + 1,
                                                self.frames[wrong_frame_id])

            res = self.engine.handle_request(mistake_msg)

            self.assertEqual(res.frame_id, i + 1)
            self.assertEqual(res.status,
                             GabrielOutput.Status.TASK_ERROR,  # human mistake
                             msg=f'{res}')

            # now fix the error by reverting to the previous frame
            res = self.engine.handle_request(msg)
            self.assertEqual(res.frame_id, i)
            self.assertEqual(res.status,
                             GabrielOutput.Status.SUCCESS,
                             msg=f'{res}')
            self.assertEqual(len(res.results), 2)
