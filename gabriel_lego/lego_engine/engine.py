import time

import numpy as np

import zhuocv3 as zc
from image_util import preprocess_img
from lego_cv import LEGOCVError, NoBoardDetectedError, NoLEGODetectedError
from gabriel_lego.lego_engine import config, protocol as proto, tasks
from gabriel_lego.lego_engine import BoardState, EmptyBoardState, \
    NoStateChangeError

from gabriel_server_common.



class LEGOEngine:
    def __init__(self, task: tasks.TaskManager):
        self.task = task
        self.state_repeat_count = 0
        self.state_change_time = time.time()

    def handle_image(self, img: np.ndarray) -> tasks.Guidance:
        try:
            state = BoardState(preprocess_img(img))
        except NoLEGODetectedError:
            state = EmptyBoardState()

        try:
            self.task.update_state(state)
            return self.task.get_guidance()
        except NoStateChangeError as e:
            self.state_repeat_count += 1
            if self.state_repeat_count >= config.BM_WINDOW_MIN_COUNT \
                    or time.time() - self.state_change_time >= \
                    config.BM_WINDOW_MIN_TIME:
                # resend previous guidance
                self.state_repeat_count = 0
                self.state_change_time = time.time()
                return self.task.get_guidance()
            else:
                raise e

    def handle_request(self, from_client: proto.FromClient) -> proto.FromServer:
        response = proto.FromServer()
        response.frame_id = from_client.frame_id
        try:
            assert from_client.type == GabrielInput.Type.IMAGE
            guidance = self.handle_image(zc.raw2cv_image(from_client.payload))

            if guidance.success:
                # no errors, either in engine or in task
                response.status = GabrielOutput.Status.SUCCESS
            else:
                # no engine error, but there is a task error
                # i.e. human has made a mistake!
                response.status = GabrielOutput.Status.TASK_ERROR

            # error or not, we still add the guidance (if included!)
            if guidance.image is not None:
                img_result = GabrielOutput.Result()
                img_result.type = GabrielOutput.ResultType.IMAGE
                img_result.payload = zc.cv_image2raw(guidance.image)

                response.results.append(img_result)

            # guidance always include text though
            txt_result = GabrielOutput.Result()
            txt_result.type = GabrielOutput.ResultType.TEXT
            txt_result.payload = guidance.instruction.encode('utf-8')

            response.results.append(txt_result)

        except NoStateChangeError:
            # no state change, and not enough repetitions to trigger a resend
            # of the previous guidance, so just ACK (no payload)
            response.status = GabrielOutput.Status.SUCCESS

        except NoBoardDetectedError:
            # most basic error, board was not detected in the image
            # this is not a task error (i.e. user mistake) but rather an
            # engine error!
            response.status = GabrielOutput.Status.ENGINE_ERROR
            result = GabrielOutput.Result()
            result.type = GabrielOutput.ResultType.TEXT
            result.payload = 'Failed to detect LEGO board in image.' \
                .encode('utf-8')
            response.results.append(result)

        except NoLEGODetectedError:
            # Detected the board, but could not detect any LEGO pieces
            # should NEVER be thrown at this level, but leaving this except
            # clause to be explicit.
            raise

        except LEGOCVError:
            # another CV error. Also an engine error.
            response.status = GabrielOutput.Status.ENGINE_ERROR
            result = GabrielOutput.Result()
            result.type = GabrielOutput.ResultType.TEXT
            result.payload = 'LEGO CV Error!'.encode('utf-8')
            response.results.append(result)

        return response
