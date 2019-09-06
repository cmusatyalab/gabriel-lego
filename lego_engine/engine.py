from typing import Optional

import cv.bitmap as bm
from cv.image_util import ImageProcessError, preprocess_img
from lego_engine import tasks
from lego_engine.protocol import *


class LEGOEngine:
    def __init__(self, task: tasks.Task):
        self.task = task
        self.just_started = True
        self.prev_bitmap = None

    def handle_image(self, img) -> Optional[tasks.Guidance]:
        # this methods raises an ImageProcessError if the image could not be
        # processed correctly

        # Todo: resend guidance

        if self.just_started:
            # if first run, send initial guidance
            self.just_started = False
            return self.task.get_guidance()

        bitmap = preprocess_img(img)
        if self.prev_bitmap is not None and \
                not bm.bitmap_same(self.prev_bitmap, bitmap):
            # changed state, do something
            self.prev_bitmap = bitmap
            self.task.update_state(bitmap)
            return self.task.get_guidance()

    def handle_request(self, proto: GabrielInput) -> GabrielOutput:
        response = GabrielOutput()
        response.frame_id = proto.frame_id
        try:
            assert proto.type == GabrielInput.Type.IMAGE
            guidance = self.handle_image(proto.payload)

            # todo parse guidance into protobuf


        except ImageProcessError:
            response.status = GabrielOutput.Status.ERROR
            result = GabrielOutput.Result()
            result.type = GabrielOutput.ResultType.TEXT
            result.payload = 'Failed to detect LEGO board in image' \
                .encode('utf-8')
            response.results = [result]

        return response
