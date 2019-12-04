import logging

import cv2
import numpy as np
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine

from gabriel_lego.cv import image_util as img_util
from gabriel_lego.cv.lego_cv import LEGOCVError, LowConfidenceError, \
    NoBoardDetectedError, NoLEGODetectedError
from gabriel_lego.lego_engine.board import BoardState, EmptyBoardState
from gabriel_lego.lego_engine.task_manager import CorrectTaskState, \
    IncorrectTaskState, NoStateChangeError
from gabriel_lego.protocol import lego_proto


class LEGOCognitiveEngine(cognitive_engine.Engine):

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tasks = []

    def handle(self, from_client):
        if from_client.payload_type != gabriel_pb2.PayloadType.IMAGE:
            return cognitive_engine.wrong_input_format_message(
                from_client.frame_id)

        # build the engine state
        engine_state = cognitive_engine.unpack_engine_fields(
            lego_proto.LEGOState,
            from_client)

        c_task = self._tasks[engine_state.task_id]
        current_state_id = engine_state.current_state_index
        t_state_id = engine_state.target_state_index

        if current_state_id >= 0:
            current_state = CorrectTaskState(c_task, current_state_id)
        else:
            p_board_state = BoardState(
                np.asarray(bytearray(engine_state.previous_error_state),
                           dtype=np.int8)
            )

            current_state = IncorrectTaskState(c_task, t_state_id,
                                               p_board_state)

        # process input
        img_array = np.asarray(bytearray(from_client.payload), dtype=np.int8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        try:
            try:
                board_state = BoardState(img_util.preprocess_img(img))
            except NoLEGODetectedError:
                # board is in frame, but it's empty
                board_state = EmptyBoardState()

            # *actually* compute next state
            next_state = current_state.compute_next_task_state(board_state)
            if next_state.is_correct:
                result = lego_proto.LEGOState.FRAME_RESULT.SUCCESS
            else:
                result = lego_proto.LEGOState.FRAME_RESULT.TASK_ERROR

        except LowConfidenceError:
            self._logger.warning('Low confidence in LEGO reconstruction.')
            result = lego_proto.LEGOState.FRAME_RESULT.LOW_CONFIDENCE_RECON
        except NoBoardDetectedError:
            # junk frame, no board in frame
            self._logger.warning('No board detected in frame.')
            result = lego_proto.LEGOState.FRAME_RESULT.JUNK_FRAME
        except LEGOCVError:
            # other CV error
            self._logger.warning('CV processing failed.')
            result = lego_proto.LEGOState.FRAME_RESULT.OTHER_CV_ERROR
        except NoStateChangeError:
            self._logger.warning('No change from previous input.')
            result = lego_proto.LEGOState.FRAME_RESULT.NO_CHANGE
