#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#


import json
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np

if os.path.isdir("../../gabriel/server"):
    sys.path.insert(0, "../../gabriel/server")
# import gabriel

# LOG = gabriel.logging.getLogger(__name__)

from . import bitmap as bm
from . import config
from . import lego_cv as lc

sys.path.insert(0, "..")
from cv import zhuocv3 as zc

# from tasks.task_Turtle import bitmaps
# from tasks.task_Turtle import time_estimates
time_estimates = None
from .tasks.task_generator import DefaultGenerator
from .tasks import ExTask

config.setup(is_streaming=True)
display_list = config.DISPLAY_LIST

LOG_TAG = "Lego Server: "


class LegoHandler:  # gabriel.network.CommonHandler):
    def setup(self):
        # LOG.info(LOG_TAG + "proxy connected to Lego server")
        super(LegoHandler, self).setup()

        self.stop = threading.Event()

        self.is_first_frame = True
        self.commited_bitmap = np.zeros((1, 1), np.int)  # basically nothing
        self.temp_bitmap = {'start_time': None, 'bitmap': None, 'count': 0}
        self.task = ExTask.ExTask(DefaultGenerator.generate(40))
        if time_estimates is not None:
            self.task.update_time_estimates(time_estimates)
        self.counter = {'confident'     : 0,
                        'not_confident' : 0,
                        'same_as_prev'  : 0,
                        'diff_from_prev': 0,
                        }

    def __repr__(self):
        return "Lego Handler"

    def _handle_input_data(self):
        img_size = struct.unpack("!I", self._recv_all(4))[0]
        img = self._recv_all(img_size)
        cv_img = zc.raw2cv_image(img)
        return_data = self._handle_img(cv_img)

        packet = struct.pack("!I%ds" % len(return_data), len(return_data),
                             return_data)
        self.request.sendall(packet)
        self.wfile.flush()

    def _handle_img(self, img):

        if self.is_first_frame and not config.RECOGNIZE_ONLY:  # do something
            # special when the task begins
            result, img_guidance = self.task.get_first_guidance()
            zc.check_and_display('guidance', img_guidance, display_list,
                                 wait_time=config.DISPLAY_WAIT_TIME,
                                 resize_max=config.DISPLAY_MAX_PIXEL)
            self.is_first_frame = False
            result['state_index'] = 0  # first step
            return json.dumps(result)

        result = {
            'status': 'nothing'
        }  # default

        stretch_ratio = float(16) / 9 * img.shape[0] / img.shape[1]
        if img.shape != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3):
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
                             interpolation=cv2.INTER_AREA)

        ## get bitmap for current image
        zc.check_and_display('input', img, display_list,
                             wait_time=config.DISPLAY_WAIT_TIME,
                             resize_max=config.DISPLAY_MAX_PIXEL)
        rtn_msg, bitmap = lc.process(img, stretch_ratio, display_list)
        if rtn_msg['status'] != 'success':
            print((rtn_msg['message']))
            if rtn_msg[
                'message'] == "Not confident about reconstruction, maybe too " \
                              "much noise":
                self.counter['not_confident'] += 1
            return json.dumps(result)

        self.counter['confident'] += 1

        ## try to commit bitmap
        state_change = False
        if bm.bitmap_same(self.commited_bitmap, bitmap):
            pass
        else:
            current_time = time.time()
            if not bm.bitmap_same(self.temp_bitmap['bitmap'], bitmap):
                self.temp_bitmap['bitmap'] = bitmap
                self.temp_bitmap['first_time'] = current_time
                self.temp_bitmap['count'] = 0
                self.counter['diff_from_prev'] += 1
            else:
                self.counter['same_as_prev'] += 1
            self.temp_bitmap['count'] += 1
            if current_time - self.temp_bitmap[
                'first_time'] > config.BM_WINDOW_MIN_TIME or self.temp_bitmap[
                'count'] >= config.BM_WINDOW_MIN_COUNT:
                self.commited_bitmap = self.temp_bitmap['bitmap']
                state_change = True
        # print "\n\n\n\n\n%s\n\n\n\n\n" % self.counter

        bitmap = self.commited_bitmap
        if 'lego_syn' in display_list and bitmap is not None:
            img_syn = bm.bitmap2syn_img(bitmap)
            zc.display_image('lego_syn', img_syn,
                             wait_time=config.DISPLAY_WAIT_TIME,
                             resize_scale=50)

        if config.RECOGNIZE_ONLY:
            return json.dumps(result)

        ## now user has done something, provide some feedback
        img_guidance = None
        if state_change:
            self.task.update_state(bitmap)
            result, img_guidance = self.task.get_guidance()

            if self.task.is_final_state():
                step_idx = len(self.task.states)
            else:
                # get current step
                step_idx = self.task.state2idx(self.task.current_state)
                # make sure step index is always -1 in case of error
                # also, differentiate from the default initial step (which we
                # assign a step index 0)
                # from the internal step index obtained from the task (which
                # also begins at 0) by
                # shifting the index by 1:
                step_idx = -1 if step_idx < 0 else step_idx + 1
            result['state_index'] = step_idx

        if img_guidance is not None:
            zc.check_and_display('guidance', img_guidance, display_list,
                                 wait_time=config.DISPLAY_WAIT_TIME,
                                 resize_max=config.DISPLAY_MAX_PIXEL)

        return json.dumps(result)

# class LegoServer(gabriel.network.CommonServer):
#     def __init__(self, port, handler):
#         gabriel.network.CommonServer.__init__(self, port,
#                                               handler)  # cannot use super
#         # because it's old style class
#         LOG.info(LOG_TAG + "* Lego server configuration")
#         LOG.info(
#             LOG_TAG + " - Open TCP Server at %s" % (str(self.server_address)))
#         LOG.info(LOG_TAG + " - Disable nagle (No TCP delay)  : %s" %
#                  str(self.socket.getsockopt(socket.IPPROTO_TCP,
#                                             socket.TCP_NODELAY)))
#         LOG.info(LOG_TAG + "-" * 50)
#
#     def terminate(self):
#         gabriel.network.CommonServer.terminate(self)
#
#
# if __name__ == "__main__":
#     lego_server = LegoServer(config.TASK_SERVER_PORT, LegoHandler)
#     lego_thread = threading.Thread(target=lego_server.serve_forever)
#     lego_thread.daemon = True
#
#     try:
#         lego_thread.start()
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt as e:
#         LOG.info(LOG_TAG + "Exit by user\n")
#         sys.exit(1)
#     except Exception as e:
#         LOG.error(str(e))
#         sys.exit(1)
#     finally:
#         if lego_server is not None:
#             lego_server.terminate()
