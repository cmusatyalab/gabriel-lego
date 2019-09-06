import logging
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
                self.logger.debug(f'Loading {frame_file} into memory...')
                self.frames.append(f.read())

        with open('./cv_bad_frame.jpeg', 'rb') as f:
            self.bad_frame = f.read()

        self.engine = LEGOEngine(Task(task_Turtle.bitmaps))

    @staticmethod
    def prepare_protobuf(frame_id: int, image: bytes) -> GabrielInput:
        proto = GabrielInput()
        proto.type = GabrielInput.Type.IMAGE
        proto.frame_id = frame_id
        proto.payload = image
        return proto

    def test_correct_run(self):
        for i, frame in enumerate(self.frames):
            msg = self.prepare_protobuf(i, frame)
            res = self.engine.handle_request(msg)

            self.assertEqual(res.frame_id, i)
            self.assertEqual(res.status,
                             GabrielOutput.Status.SUCCESS,
                             msg=f'{res}')
            self.assertEqual(len(res.results), 2)

    def test_run_with_mistakes(self):
        def correct_run():
            for i, frame in enumerate(self.frames):
                msg = self.prepare_protobuf(i, frame)
                res = self.engine.handle_request(msg)

                self.assertEqual(res.frame_id, i)
                self.assertEqual(res.status,
                                 GabrielOutput.Status.SUCCESS,
                                 msg=f'{res}')
                self.assertEqual(len(res.results), 2)
                yield msg

        for step_msg in correct_run():
            # insert error after step
            error_msg = self.prepare_protobuf(500, self.bad_frame)
            res = self.engine.handle_request(error_msg)

            self.assertEqual(res.frame_id, 500)
            self.assertEqual(res.status,
                             GabrielOutput.Status.ERROR,
                             msg=f'{res}')
            self.assertEqual(len(res.results), 1)

            # resend previous correct input
            res = self.engine.handle_request(step_msg)
            self.assertEqual(res.frame_id, step_msg.frame_id)
            self.assertEqual(res.status,
                             GabrielOutput.Status.SUCCESS,
                             msg=f'{res}')
            self.assertEqual(len(res.results), 2)
