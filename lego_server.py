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

if os.path.isdir("../../gabriel"):
    sys.path.insert(0, "../../gabriel")

from gabriel.proxy.common import LOG

# Lego related
import lego_cv as lc
import bitmap as bm
import lego_config as config
from tasks.task_Turtle import bitmaps
from tasks import Task

config.setup(is_streaming = True)
lc.set_config(is_streaming = True)

LEGO_PORT = 6090
LOG_TAG = "LEGO: "
DISPLAY_LIST = config.DISPLAY_LIST_STREAM

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
        self.target = self.task.get_state(0)

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
        cv_img = lc.raw2cv_image(img)
        return_data = self._handle_img(cv_img)
        
        packet = struct.pack("!I%ds" % len(return_data), len(return_data), return_data)
        sock.sendall(packet)

    def _handle_img(self, img):
        if self.is_first_frame: # do something special
            img_guidance = bm.bitmap2guidance_img(self.target, None)
            lc.check_and_display('guidance', img_guidance, DISPLAY_LIST, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)
            rtn_message = "Welcome to the Lego task. As a first step, please find a piece of 1x%d %s brick and put it on the board." % (self.target.shape[1], config.COLOR_ORDER[self.target[0, 0]])
            self.is_first_frame = False
            result = {'message' : rtn_message, 
                      'target' : self.target.tolist()}
            return json.dumps(result)

        result = {'message' : "nothing"} # default

        if img.shape != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3):
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

        ## get bitmap for current image
        lc.check_and_display('input', img, DISPLAY_LIST, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)
        rtn_msg, objects = lc.find_lego(img, DISPLAY_LIST)
        if objects is not None:
            img_lego, img_lego_full, img_board, img_board_ns, perspective_mtx = objects
        if rtn_msg['status'] != 'success':
            print rtn_msg['message']
            return json.dumps(result)
        rtn_msg, objects = lc.correct_orientation(img_lego, img_lego_full, DISPLAY_LIST)
        if objects is not None:
            img_lego_correct, img_lego_full_correct, rotation_mtx = objects
        if rtn_msg['status'] != 'success':
            print rtn_msg['message']
            return json.dumps(result)
        rtn_msg, bitmap = lc.reconstruct_lego(img_lego, img_board, img_board_ns, rotation_mtx, DISPLAY_LIST)
        if rtn_msg['status'] != 'success':
            print rtn_msg['message']
            return json.dumps(result)

        ## try to commit bitmap
        state_change = False
        if bm.bitmap_same(self.commited_bitmap, bitmap):
            pass
        else:
            current_time = time.time()
            if bm.bitmap_same(self.temp_bitmap['bitmap'], bitmap):
                self.temp_bitmap['count'] += 1
                if current_time - self.temp_bitmap['first_time'] > 0.1 or self.temp_bitmap['count'] > 2:
                    self.commited_bitmap = self.temp_bitmap['bitmap']
                    state_change = True
            else:
                self.temp_bitmap['bitmap'] = bitmap
                self.temp_bitmap['first_time'] = current_time
                self.temp_bitmap['count'] = 1

        if config.OPT_WINDOW:
            bitmap = self.commited_bitmap
        else:
            state_change = True
        if 'lego_syn' in DISPLAY_LIST and bitmap is not None:
            img_syn = bm.bitmap2syn_img(bitmap)
            lc.display_image('lego_syn', img_syn, wait_time = config.DISPLAY_WAIT_TIME, resize_scale = 50, save_image = config.SAVE_IMAGE)

        ## now user has done something, provide some feedback
        result = {'message' : "nothing",
                  'target' : self.target.tolist()}
        img_guidance = None
        if state_change:
            if bm.bitmap_same(bitmap, self.target):
                if self.task.is_final_state():
                    result['message'] = "You have completed the task. Congratulations!"
                    #TODO: stop the task
                else:
                    self.target = self.task.next_state()
                    result['target'] = self.target.tolist()
                    target_diff = bm.bitmap_diff(bitmap, self.target)
                    if target_diff['n_diff_pieces'] != 1:
                        result['message'] = "The task is not well designed. Now stop and check!"
                    else:
                        result['message'] = bm.generate_message(bitmap, self.target, target_diff)
                        img_guidance = bm.bitmap2guidance_img(self.target, target_diff['first_piece'])
                        result['diff_piece'] = target_diff['first_piece']
            else:
                target_diff = bm.bitmap_diff(bitmap, self.target)
                if target_diff is not None and target_diff['larger'] == 2 and target_diff['n_diff_pieces'] == 1:
                    result['message'] = bm.generate_message(bitmap, self.target, target_diff)
                    img_guidance = bm.bitmap2guidance_img(self.target, target_diff['first_piece'])
                    new_piece = target_diff['first_piece']
                else:
                    result['message'] = "This is incorrect. Now undo the last step"
                    #TODO:
        if img_guidance is not None:
            lc.check_and_display('guidance', img_guidance, DISPLAY_LIST, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)

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

