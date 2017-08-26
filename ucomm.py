#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#
#   Author: Kiryong Ha <krha@cmu.edu>
#           Zhuo Chen <zhuoc@cs.cmu.edu>
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

from base64 import b64encode, b64decode
import json
import numpy as np
from optparse import OptionParser
import os
import pprint
import Queue
import socket
import sys
import threading
import time

sys.path.insert(0, "..")
import zhuocv as zc

import bitmap as bm
import config
from tasks.task_Turtle import bitmaps
from tasks import Task

if os.path.isdir("../../gabriel/server"):
    sys.path.insert(0, "../../gabriel/server")
import gabriel
import gabriel.ucomm
LOG = gabriel.logging.getLogger(__name__)

config.setup(is_streaming = True)
display_list = config.DISPLAY_LIST

class UCommError(Exception):
    pass


class LegoResultForwardingClient(gabriel.ucomm.ResultForwardingClientBase):
    def __init__(self, control_address):
        gabriel.ucomm.ResultForwardingClientBase.__init__(self, control_address)

        self.is_first_frame = True
        self.commited_bitmap = np.zeros((1, 1), np.int) # basically nothing
        self.temp_bitmap = {'first_time' : None, 'bitmap' : None, 'count' : 0}
        self.task = Task.Task(bitmaps)


    def _generate_guidance(self, header, state, engine_id):
        if config.RECOGNIZE_ONLY:
            return json.dumps(result)

        if self.is_first_frame: # do something special when the task begins
            result, img_guidance = self.task.get_first_guidance()
            result['image'] = b64encode(zc.cv_image2raw(img_guidance))
            zc.check_and_display('guidance', img_guidance, display_list, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL)
            self.is_first_frame = False
            header['status'] = result.pop('status')
            result.pop('animation', None)
            return json.dumps(result)

        header['status'] = "success"
        result = {} # default

        if state == "None":
            header['status'] = "nothing"
            return json.dumps(result)

        bitmap = np.array(json.loads(state))

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
            self.temp_bitmap['count'] += 1
            if current_time - self.temp_bitmap['first_time'] > config.BM_WINDOW_MIN_TIME or self.temp_bitmap['count'] >= config.BM_WINDOW_MIN_COUNT:
                self.commited_bitmap = self.temp_bitmap['bitmap']
                state_change = True

        bitmap = self.commited_bitmap
        if 'lego_syn' in display_list and bitmap is not None:
            img_syn = bm.bitmap2syn_img(bitmap)
            zc.display_image('lego_syn', img_syn, wait_time = config.DISPLAY_WAIT_TIME, resize_scale = 50)

        ## now user has done something, provide some feedback
        img_guidance = None
        if state_change:
            self.task.update_state(bitmap)
            result, img_guidance = self.task.get_guidance()
            result['image'] = b64encode(zc.cv_image2raw(img_guidance))
            header['status'] = result.pop('status')
            result.pop('animation', None)

        if img_guidance is not None:
            zc.check_and_display('guidance', img_guidance, display_list, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL)

        return json.dumps(result)


def register_ucomm(ip_addr, port, net_interface = "eth0"):
    url = "http://%s:%d/" % (ip_addr, port)
    json_info = {
        gabriel.ServiceMeta.UCOMM_SERVER_IP: gabriel.network.get_ip(net_interface),
        gabriel.ServiceMeta.UCOMM_SERVER_PORT: gabriel.Const.UCOMM_SERVER_PORT
        }
    gabriel.network.http_put(url, json_info)


def main():
    ## get service list from control server
    settings = gabriel.util.process_command_line(sys.argv[1:])
    ip_addr, port = gabriel.network.get_registry_server_address(settings.address)
    service_list = gabriel.network.get_service_list(ip_addr, port)
    LOG.info("Gabriel Server :")
    LOG.info(pprint.pformat(service_list))

    ## register the current ucomm
    try:
        register_ucomm(ip_addr, port, settings.net_interface)
    except Exception as e:
        LOG.error(str(e))
        LOG.error("failed to register UCOMM to the control")
        sys.exit(1)

    control_vm_ip = service_list.get(gabriel.ServiceMeta.UCOMM_RELAY_IP)
    control_vm_port = service_list.get(gabriel.ServiceMeta.UCOMM_RELAY_PORT)

    # result pub/sub
    try:
        LOG.info("connecting to %s:%d" % (control_vm_ip, control_vm_port))
        result_forward_thread = LegoResultForwardingClient((control_vm_ip, control_vm_port))
        result_forward_thread.isDaemon = True
    except socket.error as e:
        # do not proceed if cannot connect to control VM
        if result_forward_thread is not None:
            result_forward_thread.terminate()
        raise UCommError("Failed to connect to Control server (%s:%d)" % (control_vm_ip, control_vm_port))

    # ucomm server
    ucomm_server = gabriel.ucomm.UCommServer(gabriel.Const.UCOMM_SERVER_PORT, gabriel.ucomm.UCommServerHandler)
    ucomm_server_thread = threading.Thread(target = ucomm_server.serve_forever)
    ucomm_server_thread.daemon = True

    # run the threads
    try:
        result_forward_thread.start()
        ucomm_server_thread.start()
        while True:
            time.sleep(100)
    except KeyboardInterrupt as e:
        sys.stdout.write("Exit by user\n")
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)
    finally:
        if ucomm_server is not None:
            ucomm_server.terminate()
        if result_forward_thread is not None:
            result_forward_thread.terminate()


if __name__ == '__main__':
    main()
