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
import time
import Queue
import socket
import struct
import threading
if os.path.isdir("../../gabriel"):
    sys.path.insert(0, "../../gabriel")

from gabriel.proxy.common import AppProxyStreamingClient
from gabriel.proxy.common import AppProxyThread
from gabriel.proxy.common import AppProxyError
from gabriel.proxy.common import ResultpublishClient
from gabriel.proxy.common import Protocol_measurement
from gabriel.proxy.common import get_service_list
from gabriel.proxy.common import LOG
from gabriel.proxy.launcher import AppLauncher
from gabriel.common.config import ServiceMeta as SERVICE_META
from gabriel.common.config import Const

LEGO_PORT = 6090
LOG_TAG = "LEGO Proxy: "
APP_PATH = "./lego_server.py"

class LegoProxy(AppProxyThread):
    def __init__(self, image_queue, output_queue_list, task_server_addr, log_flag = True, app_id=None ):
        super(LegoProxy, self).__init__(image_queue, output_queue_list, app_id=app_id)
        self.log_flag = log_flag
        try:
            self.task_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.task_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.task_server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.task_server_sock.connect(task_server_addr)
        except socket.error as e:
            LOG.warning(LOG_TAG + "Failed to connect to task server at %s" % str(task_server_addr))

    def terminate(self):
        if self.task_server_sock is not None:
            self.task_server_sock.close()
        super(LegoProxy, self).terminate()

    @staticmethod
    def _recv_all(socket, recv_size):
        data = ''
        while len(data) < recv_size:
            tmp_data = socket.recv(recv_size - len(data))
            if tmp_data == None or len(tmp_data) == 0:
                raise AppProxyError("Socket is closed")
            data += tmp_data
        return data

    def handle(self, header, data):
        # receive data from control VM
        LOG.info("received new image")

        # feed data to the task assistance app
        packet = struct.pack("!I%ds" % len(data), len(data), data)
        self.task_server_sock.sendall(packet)
        result_size = struct.unpack("!I", self._recv_all(self.task_server_sock, 4))[0]
        result_data = self._recv_all(self.task_server_sock, result_size)
        LOG.info("result : %s" % result_data)

        # always return result to measure the FPS
        return result_data

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80)) 
    return s.getsockname()[0]

if __name__ == "__main__":
    result_queue = list()

    sys.stdout.write("Discovering Control VM\n")
    service_list = get_service_list(sys.argv)
    video_ip = service_list.get(SERVICE_META.VIDEO_TCP_STREAMING_ADDRESS)
    video_port = service_list.get(SERVICE_META.VIDEO_TCP_STREAMING_PORT)
    return_addresses = service_list.get(SERVICE_META.RESULT_RETURN_SERVER_LIST)

    # task assistance app thread
    app_thread = AppLauncher(APP_PATH, is_print=True)
    app_thread.start()
    app_thread.isDaemon = True
    time.sleep(2)

    # image receiving thread
    video_frame_queue = Queue.Queue(Const.APP_LEVEL_TOKEN_SIZE)
    print "TOKEN SIZE OF OFFLOADING ENGINE: %d" % Const.APP_LEVEL_TOKEN_SIZE
    video_streaming = AppProxyStreamingClient((video_ip, video_port), video_frame_queue)
    video_streaming.start()
    video_streaming.isDaemon = True

    # proxy that talks with task assistance server
    task_server_ip = get_local_ip()
    task_server_port = LEGO_PORT
    app_proxy = LegoProxy(video_frame_queue, result_queue, (task_server_ip, task_server_port), log_flag = True,\
            app_id=None) # TODO: what is this id for?
    app_proxy.start()
    app_proxy.isDaemon = True

    # result pub/sub
    result_pub = ResultpublishClient(return_addresses, result_queue)
    result_pub.start()
    result_pub.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        sys.stdout.write("user exits\n")
    finally:
        if video_streaming is not None:
            video_streaming.terminate()
        if app_proxy is not None:
            app_proxy.terminate()
        result_pub.terminate()

