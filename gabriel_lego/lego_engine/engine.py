import logging
import time

import cv2
import numpy as np
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine

from gabriel_lego.cv import image_util as img_util
from gabriel_lego.cv.lego_cv import LEGOCVError, LowConfidenceError, \
    NoBoardDetectedError, NoLEGODetectedError
from gabriel_lego.lego_engine.board import BoardState, EmptyBoardState
from gabriel_lego.lego_engine.task_manager import CorrectTaskState, \
    DefaultTaskManager, IncorrectTaskState, NoStateChangeError, NoSuchTaskError
from gabriel_lego.protocol import lego_proto


class LEGOCognitiveEngine(cognitive_engine.Engine):
    _DEFAULT_TASK = 'turtle_head'

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def handle(self, from_client):
        received = time.time()

        if from_client.payload_type != gabriel_pb2.PayloadType.IMAGE:
            return cognitive_engine.wrong_input_format_message(
                from_client.frame_id)

        # build the engine state
        old_lego_state = cognitive_engine.unpack_engine_fields(
            lego_proto.LEGOState,
            from_client)

        try:
            c_task = DefaultTaskManager.get_task(old_lego_state.task_id)
        except NoSuchTaskError:
            self._logger.warning(f'Invalid task name: {old_lego_state.task_id}')
            self._logger.warning(f'Running with default task instead: '
                                 f'{self._DEFAULT_TASK}')
            c_task = DefaultTaskManager.get_task(self._DEFAULT_TASK)

        current_state_id = old_lego_state.current_state_index
        t_state_id = old_lego_state.target_state_index

        if current_state_id >= 0:
            current_state = CorrectTaskState(c_task, current_state_id)
        else:
            p_board_state = BoardState(
                np.asarray(bytearray(old_lego_state.previous_error_state),
                           dtype=np.int8)
            )

            current_state = IncorrectTaskState(c_task, t_state_id,
                                               p_board_state)

        # process input
        img_array = np.asarray(bytearray(from_client.payload), dtype=np.int8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        result_wrapper = gabriel_pb2.ResultWrapper()
        new_lego_state = lego_proto.LEGOState()

        result_wrapper.frame_id = from_client.frame_id
        result_wrapper.status = gabriel_pb2.ResultWrapper.Status.SUCCESS

        new_lego_state.target_state_index = old_lego_state.target_state_index
        new_lego_state.current_state_index = old_lego_state.current_state_index

        try:
            try:
                board_state = BoardState(img_util.preprocess_img(img))
            except NoLEGODetectedError:
                # board is in frame, but it's empty
                board_state = EmptyBoardState()

            # *actually* compute next state
            next_state = current_state.compute_next_task_state(board_state)

            if next_state.is_correct:
                # step was performed correctly!

                new_lego_state.result = \
                    lego_proto.LEGOState.FRAME_RESULT.SUCCESS

                if next_state.is_final:
                    new_lego_state.task_finished = True
            else:
                # task error
                new_lego_state.result = \
                    lego_proto.LEGOState.FRAME_RESULT.TASK_ERROR

                new_lego_state.error_prev_board_state = \
                    board_state.bitmap.tobytes()

            new_lego_state.current_state_index = next_state.state_index
            new_lego_state.target_state_index = next_state.next_state_index

            # get guidance and put it into the wrapper
            img_result = gabriel_pb2.ResultWrapper.Result()
            img_result.type = gabriel_pb2.PayloadType.IMAGE
            img_result.payload = cv2.imencode('.jpg', next_state.current_image)

            txt_result = gabriel_pb2.ResultWrapper.Result()
            txt_result.type = gabriel_pb2.PayloadType.TEXT
            txt_result.payload = next_state.current_instruction.encode('utf-8')

            result_wrapper.results.append(img_result)
            result_wrapper.results.append(txt_result)

        except LowConfidenceError:
            self._logger.warning('Low confidence in LEGO reconstruction.')
            new_lego_state.result = \
                lego_proto.LEGOState.FRAME_RESULT.LOW_CONFIDENCE_RECON
        except NoBoardDetectedError:
            # junk frame, no board in frame
            self._logger.warning('No board detected in frame.')
            new_lego_state.result = lego_proto.LEGOState.FRAME_RESULT.JUNK_FRAME
        except LEGOCVError:
            # other CV error
            self._logger.warning('CV processing failed.')
            new_lego_state.result = \
                lego_proto.LEGOState.FRAME_RESULT.OTHER_CV_ERROR
            result_wrapper.status = \
                gabriel_pb2.ResultWrapper.Status.ENGINE_ERROR
        except NoStateChangeError:
            self._logger.warning('No change from previous input.')
            new_lego_state.result = lego_proto.LEGOState.FRAME_RESULT.NO_CHANGE

        # finalize the LEGO state
        new_lego_state.task_id = old_lego_state.task_id

        timestamps = lego_proto.LEGOState.Timestamps()
        timestamps.received = received
        timestamps.sent = time.time()
        new_lego_state.timestamps = timestamps

        # finally, pack the LEGO state into the ResultWrapper
        result_wrapper.engine_fields.Pack(new_lego_state)

        return result_wrapper
