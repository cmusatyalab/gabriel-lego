from typing import Optional

import numpy as np

from cv import zhuocv3 as zc
from cv.image_util import preprocess_img
from cv.lego_cv import LEGOCVError
from lego_engine import tasks
from lego_engine.protocol import *


class LEGOEngine:
    def __init__(self, task: tasks.Task):
        self.task = task
        self.just_started = True
        # self.prev_bitmap = None

    def handle_image(self, img: np.ndarray) -> Optional[tasks.Guidance]:
        # this methods raises an ImageProcessError if the image could not be
        # processed correctly

        # Todo: resend guidance

        if self.just_started:
            # if first run, send initial guidance
            self.just_started = False
            return self.task.get_guidance()

        bitmap = preprocess_img(img)
        # changed state, do something
        # self.prev_bitmap = bitmap

        self.task.update_state(bitmap)
        return self.task.get_guidance()

    def handle_request(self, proto: GabrielInput) -> GabrielOutput:
        response = GabrielOutput()
        response.frame_id = proto.frame_id
        try:
            assert proto.type == GabrielInput.Type.IMAGE
            guidance = self.handle_image(zc.raw2cv_image(proto.payload))

            response.status = GabrielOutput.Status.SUCCESS

            img_result = GabrielOutput.Result()
            img_result.type = GabrielOutput.ResultType.IMAGE
            img_result.payload = zc.cv_image2raw(guidance.image)

            txt_result = GabrielOutput.Result()
            txt_result.type = GabrielOutput.ResultType.TEXT
            txt_result.payload = guidance.instruction.encode('utf-8')

            response.results.extend([img_result, txt_result])

        except LEGOCVError:  # todo: disambiguate into different cv errors
            response.status = GabrielOutput.Status.ERROR
            result = GabrielOutput.Result()
            result.type = GabrielOutput.ResultType.TEXT
            result.payload = 'Failed to detect LEGO board in image.' \
                .encode('utf-8')
            response.results.append(result)

        return response
