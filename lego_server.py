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
import cv2 # TODO: intend to implement all real stuff in another function
import time
import select
import socket
import struct
import threading
import traceback
import numpy as np
import lego_cv as lc
import lego_config as config
if os.path.isdir("../../../"):
    sys.path.insert(0, "../../../")

from gabriel.proxy.common import LOG

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
        img = cv_img
        if img.shape != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3):
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

        # tempory: prepare a null packet
        return_data = "nothing"
        packet = struct.pack("!I%ds" % len(return_data), len(return_data), return_data)

        if 'input' in DISPLAY_LIST:
            lc.display_image('input', img)
        rtn_msg, img_lego, perspective_mtx = lc.locate_lego(img, DISPLAY_LIST)
        if rtn_msg['status'] != 'success':
            print rtn_msg['message']
            sock.sendall(packet)
            return
        rtn_msg, img_lego_correct = lc.correct_orientation(img_lego, perspective_mtx, DISPLAY_LIST)
        if rtn_msg['status'] != 'success':
            sock.sendall(packet)
            return
        rtn_msg, bitmap = lc.reconstruct_lego(img_lego_correct, DISPLAY_LIST)
        if rtn_msg['status'] != 'success':
            sock.sendall(packet)
            return
        if 'lego_syn' in DISPLAY_LIST:
            img_syn = lc.bitmap2syn_img(bitmap)
            lc.display_image('lego_syn', img_syn)

        sock.sendall(packet)
        
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

