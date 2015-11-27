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

import os
import sys
import cv2
import time
import json
import select
import socket
import struct
import threading
import traceback
import numpy as np
from base64 import b64encode, b64decode

if os.path.isdir("../../gabriel"):
    sys.path.insert(0, "../../gabriel")

from gabriel.proxy.common import LOG
from gabriel.common.config import DEBUG
from gabriel.proxy.common import Protocol_measurement

# Lego related
sys.path.insert(0, "..")
import bitmap as bm
import config
import lego_cv as lc
import zhuocv as zc
from tasks.task_Turtle import bitmaps
from tasks import Task

config.setup(is_streaming = True)
lc.set_config(is_streaming = True)

LEGO_PORT = 6090
LOG_TAG = "LEGO: "
DISPLAY_LIST = config.DISPLAY_LIST

class LegoProcessing(threading.Thread):
    def __init__(self):
        self.stop = threading.Event()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server.bind(("", LEGO_PORT))
        self.server.listen(10) # actually we are only expecting one connection...

        self.is_first_frame = True
        self.commited_bitmap = np.zeros((1, 1), np.int) # basically nothing
        self.temp_bitmap = {'start_time' : None, 'bitmap' : None, 'count' : 0}
        self.task = Task.Task(bitmaps)
        self.counter = {'confident'      : 0,
                        'not_confident'  : 0,
                        'same_as_prev'   : 0,
                        'diff_from_prev' : 0,
                       }

        for display_name in DISPLAY_LIST:
            cv2.namedWindow(display_name)

        threading.Thread.__init__(self, target=self.run)

    def run(self):
        input_list = [self.server]
        output_list = []
        error_list = []

        LOG.info(LOG_TAG + "Lego processing thread started")
        try:
            while(not self.stop.wait(0.001)):
                inputready, outputready, exceptready = \
                        select.select(input_list, output_list, error_list, 0.001)
                for s in inputready:
                    if s == self.server:
                        LOG.debug(LOG_TAG + "client connected")
                        client, address = self.server.accept()
                        input_list.append(client)
                        output_list.append(client)
                        error_list.append(client)
                    else:
                        self._receive(s)
        except Exception as e:
            LOG.warning(LOG_TAG + traceback.format_exc())
            LOG.warning(LOG_TAG + "%s" % str(e))
            LOG.warning(LOG_TAG + "handler raises exception")
            LOG.warning(LOG_TAG + "Server is disconnected unexpectedly")
        LOG.debug(LOG_TAG + "Lego processing thread terminated")

    @staticmethod
    def _recv_all(socket, recv_size):
        data = ''
        while len(data) < recv_size:
            tmp_data = socket.recv(recv_size - len(data))
            if tmp_data == None or len(tmp_data) == 0:
                raise Exception("Socket is closed")
            data += tmp_data
        return data

    def _receive(self, sock):
        img_size = struct.unpack("!I", self._recv_all(sock, 4))[0]
        img = self._recv_all(sock, img_size)
        cv_img = zc.raw2cv_image(img)
        return_data = self._handle_img(cv_img)

        packet = struct.pack("!I%ds" % len(return_data), len(return_data), return_data)
        sock.sendall(packet)

    def _handle_img(self, img):
        if self.is_first_frame and not config.RECOGNIZE_ONLY: # do something special when the task begins
            result_tmp, img_guidance = self.task.get_first_guidance()
            result = {'status' : "success"}
            result['speech'] = result_tmp['message']
            result['image'] = b64encode(zc.cv_image2raw(img_guidance))
            zc.check_and_display('guidance', img_guidance, DISPLAY_LIST, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)
            self.is_first_frame = False
            return json.dumps(result)

        result = {'status' : "nothing"} # default

        stretch_ratio = float(16) / 9 * img.shape[0] / img.shape[1]
        if img.shape != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3):
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

        ## get bitmap for current image
        zc.check_and_display('input', img, DISPLAY_LIST, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)
        rtn_msg, bitmap = lc.process(img, stretch_ratio, DISPLAY_LIST)
        if DEBUG.PACKET:
            result[Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = time.time()
        if rtn_msg['status'] != 'success':
            print rtn_msg['message']
            if rtn_msg['message'] == "Not confident about reconstruction, maybe too much noise":
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
            if current_time - self.temp_bitmap['first_time'] > config.BM_WINDOW_MIN_TIME or self.temp_bitmap['count'] >= config.BM_WINDOW_MIN_COUNT:
                self.commited_bitmap = self.temp_bitmap['bitmap']
                state_change = True

        bitmap = self.commited_bitmap
        if 'lego_syn' in DISPLAY_LIST and bitmap is not None:
            img_syn = bm.bitmap2syn_img(bitmap)
            zc.display_image('lego_syn', img_syn, wait_time = config.DISPLAY_WAIT_TIME, resize_scale = 50, save_image = config.SAVE_IMAGE)

        ## now user has done something, provide some feedback
        img_guidance = None
        if state_change:
            self.task.update_state(bitmap)
            result_tmp, img_guidance = self.task.get_guidance()
            result['status'] = "success"
            result['speech'] = result_tmp['message']
            result['image'] = b64encode(zc.cv_image2raw(img_guidance))

        if img_guidance is not None:
            zc.check_and_display('guidance', img_guidance, DISPLAY_LIST, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)

        return json.dumps(result)

    def terminate(self):
        self.stop.set()

if __name__ == "__main__":
    # a thread to receive incoming images
    lego_processing = LegoProcessing()
    lego_processing.start()
    lego_processing.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        sys.stdout.write("user exits\n")
    finally:
        if lego_processing is not None:
            lego_processing.terminate()

