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

import cv2
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
import lego_cv as lc
sys.path.insert(0, "..")
import zhuocv as zc

config.setup(is_streaming = True)
display_list = config.DISPLAY_LIST

LOG_TAG = "LEGO Proxy: "
APP_PATH = "./lego_server.py"
ENGINE_ID = "LEGO_SLOW"


class LegoProxy(gabriel.network.CommonClient):
    def __init__(self, master_server_addr, engine_id, log_flag = True):
        gabriel.network.CommonClient.__init__(self, master_server_addr)
        self.log_flag = log_flag

        # tell the master proxy my identity (engine_id)
        try:
            packet = struct.pack("!I%ds" % len(engine_id), len(engine_id), engine_id)
            self.sock.sendall(packet)
            # engine number is the seq number for all engines with the same engine_id
            self.engine_number = struct.unpack("!I", self._recv_all(4))[0]
            LOG.info("Engine number is: %d" % self.engine_number)
        except Exception as e:
            raise gabriel.proxy.ProxyError("Failed to send engine name to control server: %s" % e)

    def __repr__(self):
        return "Lego Proxy"

    def _handle_input_data(self):
        # receive data from control VM
        header_size = struct.unpack("!I", self._recv_all(4))[0]
        data_size = struct.unpack("!I", self._recv_all(4))[0]
        header_str = self._recv_all(header_size)
        image_data = self._recv_all(data_size)
        #header = json.loads(header_str)

        ## symbolic representation extraction
        img = zc.raw2cv_image(image_data)
        stretch_ratio = float(16) / 9 * img.shape[0] / img.shape[1]
        if img.shape != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3):
            img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
        zc.check_and_display('input', img, display_list, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL)

        # get bitmap for current image
        rtn_msg, bitmap = lc.process(img, stretch_ratio, display_list)
        if rtn_msg['status'] != 'success':
            print rtn_msg['message']
            result_str = "None"
        else:
            result_str = json.dumps(bitmap.tolist())

        packet = struct.pack("!I%dsI%ds" % (len(header_str), len(result_str)), len(header_str), header_str, len(result_str), result_str)
        self.sock.sendall(packet)


def get_service(ip_addr, port, name):
    url = "http://%s:%d/services/%s" % (ip_addr, port, name)
    content = gabriel.network.http_get(url)
    return content

if __name__ == "__main__":
    settings = gabriel.util.process_command_line(sys.argv[1:])

    ip_addr, port = gabriel.network.get_registry_server_address(settings.address)

    # get information about master node
    master_info = get_service(ip_addr, port, "LEGO_MASTER")
    try:
        master_service_content = master_info["service_content"]
        master_ip = master_service_content['ip']
        master_port = master_service_content['port']
        LOG.info("Got master proxy at %s:%d" % (master_ip, master_port))
    except KeyError:
        LOG.error("master proxy hasn't been registered\n")
        sys.exit()

    # worker proxy
    app_proxy = LegoProxy((master_ip, master_port), engine_id = ENGINE_ID)
    app_proxy.start()
    app_proxy.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        LOG.info("user exits\n")
    finally:
        if app_proxy is not None:
            app_proxy.terminate()
