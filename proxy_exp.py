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
import multiprocessing
import os
import pprint
import Queue
import socket
import struct
import sys
import threading
import time

if os.path.isdir("../../gabriel/server"):
    sys.path.insert(0, "../../gabriel/server")
import gabriel
import gabriel.proxy
LOG = gabriel.logging.getLogger(__name__)

import config

LOG_TAG = "LEGO Proxy: "
APP_PATH = "./lego_server_exp.py"

class LegoProxy(gabriel.proxy.CognitiveProcessThread):
    def __init__(self, image_queue, output_queue, task_server_addr, engine_id, log_flag = True):
        super(LegoProxy, self).__init__(image_queue, output_queue, engine_id)
        self.log_flag = log_flag
        try:
            self.task_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.task_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.task_server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.task_server_sock.connect(task_server_addr)
        except socket.error as e:
            LOG.warning(LOG_TAG + "Failed to connect to task server at %s" % str(task_server_addr))

    def __repr__(self):
        return "Lego Proxy"

    def terminate(self):
        if self.task_server_sock is not None:
            self.task_server_sock.close()
        super(LegoProxy, self).terminate()

    def _recv_all(self, sock, recv_size):
        data = ''
        while len(data) < recv_size:
            tmp_data = sock.recv(recv_size - len(data))
            if tmp_data == None:
                raise gabriel.network.TCPNetworkError("Cannot recv data at %s" % str(self))
            if len(tmp_data) == 0:
                raise gabriel.network.TCPZeroBytesError("Recv 0 bytes.")
            data += tmp_data
        return data

    def handle(self, header, data):
        # feed data to the task assistance app
        packet = struct.pack("!I%ds" % len(data), len(data), data)
        self.task_server_sock.sendall(packet)
        try:
            result_size = struct.unpack("!I", self._recv_all(self.task_server_sock, 4))[0]
            result_data = self._recv_all(self.task_server_sock, result_size)
        except gabriel.network.TCPZeroBytesError as e:
            LOG.warning("Lego server disconnedted")
            result_data = json.dumps({'status' : "nothing"})
            self.terminate()

        result_json = json.loads(result_data)
        header['status'] = result_json.pop('status')
        header[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = result_json.pop(gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME, -1)
        result_data = json.dumps(result_json)

        return result_data


if __name__ == "__main__":
    settings = gabriel.util.process_command_line(sys.argv[1:])

    ip_addr, port = gabriel.network.get_registry_server_address(settings.address)
    service_list = gabriel.network.get_service_list(ip_addr, port)
    LOG.info("Gabriel Server :")
    LOG.info(pprint.pformat(service_list))

    video_ip = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_IP)
    video_port = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_PORT)
    ucomm_ip = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_IP)
    ucomm_port = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_PORT)

    # task assistance app thread
    app_thread = gabriel.proxy.AppLauncher(APP_PATH, is_print = True)
    app_thread.start()
    app_thread.isDaemon = True
    time.sleep(2)

    # image receiving thread
    image_queue = Queue.Queue(gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    print "TOKEN SIZE OF OFFLOADING ENGINE: %d" % gabriel.Const.APP_LEVEL_TOKEN_SIZE
    video_streaming = gabriel.proxy.SensorReceiveClient((video_ip, video_port), image_queue)
    video_streaming.start()
    video_streaming.isDaemon = True

    # app proxy
    result_queue = multiprocessing.Queue()

    task_server_ip = gabriel.network.get_ip()
    task_server_port = config.TASK_SERVER_PORT
    app_proxy = LegoProxy(image_queue, result_queue, (task_server_ip, task_server_port), engine_id = "Lego")
    app_proxy.start()
    app_proxy.isDaemon = True

    # result pub/sub
    result_pub = gabriel.proxy.ResultPublishClient((ucomm_ip, ucomm_port), result_queue)
    result_pub.start()
    result_pub.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        LOG.info("user exits\n")
    finally:
        if video_streaming is not None:
            video_streaming.terminate()
        if app_proxy is not None:
            app_proxy.terminate()
        result_pub.terminate()
        if app_thread is not None:
            app_thread.terminate()

