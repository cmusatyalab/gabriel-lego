import logging
import time
from typing import Optional

import cv2
import numpy as np
from gabriel_protocol import gabriel_pb2
from gabriel_server import cognitive_engine

from gabriel_lego.cv import image_util as img_util
from gabriel_lego.cv.lego_cv import LowConfidenceError, \
    NoBoardDetectedError, NoLEGODetectedError
from gabriel_lego.lego_engine.board import BoardState, EmptyBoardState
from gabriel_lego.lego_engine.task_manager import DefaultTaskManager, \
    IncorrectTaskState, InitialTaskState, NoStateChangeError, NoSuchTaskError, \
    TaskState
from gabriel_lego.protocol import instruction_proto


class LEGOCognitiveEngine(cognitive_engine.Engine):
    _DEFAULT_TASK = 'turtle_head'

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Initializing LEGO Engine.')

    @staticmethod
    def _build_LEGO_state(task_id: str,
                          recv_time: float,
                          status: instruction_proto.LEGOState.STATE,
                          result: instruction_proto.LEGOState.FRAME_RESULT,
                          target_state: int,
                          current_state: int,
                          prev_board_state: Optional[BoardState] = None) \
            -> instruction_proto.LEGOState:

        lego_state = instruction_proto.LEGOState()
        lego_state.task_id = task_id
        lego_state.status = status
        lego_state.result = result
        lego_state.target_state_index = target_state
        lego_state.current_state_index = current_state

        if prev_board_state:
            board_state_b = prev_board_state.bitmap.tobytes()
            lego_state.error_prev_board_state = board_state_b

        timestamps = instruction_proto.LEGOState.Timestamps()
        timestamps.received = recv_time
        timestamps.sent = time.time()

        lego_state.timestamps = timestamps

        return lego_state

    @staticmethod
    def _wrap_LEGO_state(frame_id: int,
                         status: gabriel_pb2.ResultWrapper.Status,
                         lego_state: instruction_proto.LEGOState,
                         update_cnt: int,
                         img_guidance: Optional[np.ndarray] = None,
                         txt_guidance: Optional[str] = None) \
            -> gabriel_pb2.ResultWrapper:

        result = gabriel_pb2.ResultWrapper()
        result.frame_id = frame_id
        result.status = status

        engine_fields = instruction_proto.EngineFields()
        engine_fields.update_count = update_cnt
        engine_fields.lego = lego_state

        result.engine_fields.Pack(engine_fields)

        if img_guidance:
            img_result = gabriel_pb2.ResultWrapper.Result()
            img_result.type = gabriel_pb2.PayloadType.IMAGE
            img_result.payload = cv2.imencode('.jpg', img_guidance)
            result.results.append(img_result)

        if txt_guidance:
            txt_result = gabriel_pb2.ResultWrapper.Result()
            txt_result.type = gabriel_pb2.PayloadType.TEXT
            txt_result.payload = txt_guidance.encode('utf-8')
            result.results.append(txt_result)

        return result

    def handle(self, from_client):
        received = time.time()

        if from_client.payload_type != gabriel_pb2.PayloadType.IMAGE:
            self._logger.error('Wrong input type. LEGO expects inputs to be '
                               'images, but the received input is of type: '
                               f'{from_client.payload_type.__name__}')
            return cognitive_engine.wrong_input_format_message(
                from_client.frame_id)

        self._logger.info('Received a frame from client.')

        # build the engine state
        engine_fields = cognitive_engine.unpack_engine_fields(
            instruction_proto.EngineFields,
            from_client)

        old_lego_state = engine_fields.lego

        try:
            c_task = DefaultTaskManager.get_task(old_lego_state.task_id)
        except NoSuchTaskError:
            self._logger.warning(f'Invalid task name: {old_lego_state.task_id}')
            self._logger.warning(f'Proceeding with default task instead: '
                                 f'{self._DEFAULT_TASK}')
            c_task = DefaultTaskManager.get_task(self._DEFAULT_TASK)
            old_lego_state.task_id = self._DEFAULT_TASK

        current_state_id = old_lego_state.current_state_index
        target_state_id = old_lego_state.target_state_index

        if old_lego_state.status == \
                instruction_proto.LEGOState.STATUS.INIT:
            current_state = InitialTaskState(c_task)

            # immediately return initial guidance
            return LEGOCognitiveEngine._wrap_LEGO_state(
                frame_id=from_client.frame_id,
                status=gabriel_pb2.ResultWrapper.Status.SUCCESS,
                lego_state=LEGOCognitiveEngine._build_LEGO_state(
                    task_id=old_lego_state.task_id,
                    recv_time=received,
                    status=instruction_proto.LEGOState.STATUS.WAITING_FOR_BOARD,
                    result=instruction_proto.LEGOState.FRAME_RESULT.SUCCESS,
                    target_state=0,
                    current_state=-1,
                ),
                update_cnt=engine_fields.update_count + 1,
                img_guidance=current_state.current_image,
                txt_guidance=current_state.current_instruction
            )

        elif old_lego_state.status == \
                instruction_proto.LEGOState.STATUS.WAITING_FOR_BOARD:
            # initial state, waiting for LEGO board to start task
            current_state = InitialTaskState(c_task)
        elif old_lego_state.status == \
                instruction_proto.LEGOState.STATUS.NORMAL:
            # normal, non-error state
            current_state = TaskState.generate_correct_state(c_task,
                                                             current_state_id)
        elif old_lego_state.status == \
                instruction_proto.LEGOState.STATUS.ERROR:
            # task error
            p_board_state = BoardState(
                np.asarray(bytearray(old_lego_state.previous_error_state),
                           dtype=np.int8)
            )
            current_state = IncorrectTaskState(c_task,
                                               target_state_id,
                                               p_board_state)
        elif old_lego_state.status == \
                instruction_proto.LEGOState.STATUS.FINISHED:
            # finished the task, just return empty success messages ad infinitum

            return LEGOCognitiveEngine._wrap_LEGO_state(
                frame_id=from_client.frame_id,
                status=gabriel_pb2.ResultWrapper.Status.SUCCESS,
                lego_state=LEGOCognitiveEngine._build_LEGO_state(
                    task_id=old_lego_state.task_id,
                    recv_time=received,
                    status=instruction_proto.LEGOState.STATUS.FINISHED,
                    result=instruction_proto.LEGOState.FRAME_RESULT.SUCCESS,
                    target_state=-1,
                    current_state=-1,
                ),
                update_cnt=engine_fields.update_count + 1,
            )

        else:
            # unimplemented state
            # todo, send error message?
            raise RuntimeError()

        board_state = None
        img_guidance = None
        txt_guidance = None
        try:
            # process input
            img_array = np.asarray(bytearray(from_client.payload),
                                   dtype=np.int8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            # process the image into a BoardState
            try:
                board_state = BoardState(img_util.preprocess_img(img))
            except NoLEGODetectedError:
                # board is in frame, but it's empty
                self._logger.info('Detected empty board.')
                board_state = EmptyBoardState()

            # *actually* compute next state
            next_state = current_state.compute_next_task_state(board_state)

            # code after this point will only execute if board state was
            # correctly processed (previous statements all shortcut through
            # exceptions on failure)
            current_state_id = next_state.state_index
            target_state_id = next_state.next_state_index

            img_guidance = next_state.current_image
            txt_guidance = next_state.current_instruction

            if next_state.is_correct:
                # step was performed correctly!
                result = instruction_proto.LEGOState.FRAME_RESULT.SUCCESS
                if next_state.is_final:
                    self._logger.info('Finished task!')
                    status = instruction_proto.LEGOState.STATUS.FINISHED
                else:
                    self._logger.info('Step was performed correctly, providing '
                                      'guidance for next step.')
                    status = instruction_proto.LEGOState.STATUS.NORMAL

            else:
                # task error
                status = instruction_proto.LEGOState.STATUS.ERROR
                result = instruction_proto.LEGOState.FRAME_RESULT.TASK_ERROR

                self._logger.info('Input is incorrect! Providing guidance to '
                                  'correct error.')

        except LowConfidenceError:
            self._logger.warning('Low confidence in LEGO reconstruction.')
            status = instruction_proto.LEGOState.STATUS.NORMAL
            result = \
                instruction_proto.LEGOState.FRAME_RESULT.LOW_CONFIDENCE_RECON
        except NoBoardDetectedError:
            # junk frame, no board in frame
            self._logger.warning('No board detected in frame.')
            status = instruction_proto.LEGOState.STATUS.NORMAL
            result = \
                instruction_proto.LEGOState.FRAME_RESULT.JUNK_FRAME
        except NoStateChangeError:
            self._logger.warning('No change from previous input.')
            status = instruction_proto.LEGOState.STATUS.NORMAL
            result = \
                instruction_proto.LEGOState.FRAME_RESULT.NO_CHANGE
        except Exception as e:
            # anything else
            self._logger.error('CV processing failed!')
            self._logger.error(e)

            # immediately return
            return LEGOCognitiveEngine._wrap_LEGO_state(
                frame_id=from_client.frame_id,
                status=gabriel_pb2.ResultWrapper.Status.ENGINE_ERROR,
                lego_state=LEGOCognitiveEngine._build_LEGO_state(
                    task_id=old_lego_state.task_id,
                    recv_time=received,
                    status=instruction_proto.LEGOState.STATUS.ERROR,
                    result=
                    instruction_proto.LEGOState.FRAME_RESULT.OTHER_CV_ERROR,
                    target_state=-1,
                    current_state=-1,
                ),
                update_cnt=engine_fields.update_count + 1,
            )

        self._logger.info('Sending result to client...')
        return LEGOCognitiveEngine._wrap_LEGO_state(
            frame_id=from_client.frame_id,
            status=gabriel_pb2.ResultWrapper.Status.SUCCESS,
            lego_state=LEGOCognitiveEngine._build_LEGO_state(
                task_id=old_lego_state.task_id,
                recv_time=received,
                status=status,
                result=result,
                target_state=target_state_id,
                current_state=current_state_id,
                prev_board_state=board_state
            ),
            update_cnt=engine_fields.update_count + 1,
            img_guidance=img_guidance,
            txt_guidance=txt_guidance
        )
