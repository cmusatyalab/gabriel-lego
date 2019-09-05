from typing import List, NamedTuple, Optional

import cv.bitmap as bm
from cv.image_util import ImageProcessError, preprocess_img
from lego_engine import tasks


class Guidance(NamedTuple):
    success: bool
    instruction: str
    animation: Optional[List[str]]
    step_id: int


class LEGOEngine:
    def __init__(self, task: tasks.Task):
        self.task = task
        self.just_started = True
        self.prev_bitmap = None
        self.prev_guidance = None

    def handle_image(self, img) -> Optional[Guidance]:

        # Todo: resend guidance

        if self.just_started:
            # if first run, send initial guidance
            self.just_started = False
            self.prev_guidance = self.task.get_first_guidance()
            return self.prev_guidance

        try:
            bitmap = preprocess_img(img)
            if self.prev_bitmap is not None and \
                    not bm.bitmap_same(self.prev_bitmap, bitmap):
                # changed state, do something
                self.prev_bitmap = bitmap
                self.task.update_state(bitmap)
                return self.task.get_guidance()

        except ImageProcessError as e:
            # todo handle error
            return Guidance(False, str(e.args), None)
