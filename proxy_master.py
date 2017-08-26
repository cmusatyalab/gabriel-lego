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
import numpy as np
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

from tasks.task_Turtle import bitmaps
from tasks import Task

LOG_TAG = "In (Lego) Proxy: "
ENGINE_ID = "LEGO_MASTER"

class MasterProxy(gabriel.proxy.MasterProxyThread):
    def __init__(self, image_queue, worker_queue_dict, engine_id, log_flag = True):
        super(MasterProxy, self).__init__(image_queue, engine_id)
        self.image_queue_dict = worker_queue_dict
        self.log_flag = log_flag

    def __repr__(self):
        return "Master Proxy for Lego"

    def handle(self, header, data):
        ## put current image data in a registered cognitive engine queue
        ## one engine_id gets one image!
        for engine_id, engine_info in self.image_queue_dict.iteritems():
            tokens = engine_info['tokens']
            for idx, token in enumerate(tokens):
                if token > 0:
                    tokens[idx] = token - 1
                    image_queue = engine_info['queues'][idx]
                    try:
                        image_queue.put_nowait((header, data))
                    except Queue.Full as e:
                        image_queue.get_nowait()
                        image_queue.put_nowait((header, data))
                    break


class ResultFilter(gabriel.proxy.CognitiveProcessThread):
    def __init__(self, input_queue, output_queue, worker_queue_dict, engine_id, log_flag = True):
        super(ResultFilter, self).__init__(input_queue, output_queue, engine_id)
        self.image_queue_dict = worker_queue_dict
        self.log_flag = log_flag

        ## engine info initialization
        self.engine_id_list = ['LEGO_SLOW']
        self.best_engine = 'LEGO_SLOW'
        self.check_algorithm = 'last'
        self.check_TH = 5

        self.state_history = {}
        self.good_list = []
        self.is_prev_good = 0
        self.prev_best_engine_frame_id = -1

        ## Lego specific
        self.is_first_frame = True
        self.commited_bitmap = np.zeros((1, 1), np.int) # basically nothing
        self.temp_bitmap = {'first_time' : None, 'bitmap' : None, 'count' : 0}
        self.task = Task.Task(bitmaps)

    def __repr__(self):
        return "Result Filter"

    def _trust(self, state, engine_id, frame_id, keep_time = 10000):
        ## slower than the best one...
        if frame_id <= self.prev_best_engine_frame_id:
            return "too_slow"

        ## see if other engines have had results for the same frame
        frame_state_history = self.state_history.get(frame_id, None)
        if frame_state_history is None: # this is the first result
            self.state_history[frame_id] = {engine_id : state, 'RETURNED' : False}
        else:
            self.state_history[frame_id][engine_id] = state
            if self.state_history[frame_id]['RETURNED']: # slower than other engines
                return "slow"

        ## the current result is not slow, let's see how good it is...
        # clean the list
        if engine_id == self.best_engine:
            return "success"

        if self.check_algorithm == 'last':
            if self.is_prev_good >= self.check_TH:
                self.state_history[frame_id]['RETURNED'] = True
                return "success"
            else:
                return "no_trust"

        now = time.time()
        for idx, (check_state, add_time) in enumerate(self.good_list):
            if now - add_time > keep_time:
                del self.good_list[idx]

        for check_state, add_time in self.good_list:
            if bm.bitmap_same(state, check_state):
                self.state_history[frame_id]['RETURNED'] = True
                return "success"
        return "no_trust"

    def _update_good_list(self, state, frame_id):
        frame_state_history = self.state_history.get(frame_id, None)
        if frame_state_history is None: # this could only happen if the best algorithm doesn't return in order (e.g. multiple instances of it)
            return
        now = time.time()
        for en in self.engine_id_list:
            if en == self.best_engine:
                continue
            en_detected_state = frame_state_history.get(en, None)

            if bm.bitmap_same(en_detected_state, state): # the engine did well in this detection
                if self.check_algorithm == 'last':
                    self.is_prev_good += 1
                    return
                found = False
                for idx, (check_state, add_time) in enumerate(self.good_list):
                    if bm.bitmap_same(check_state, state):
                        self.good_list[idx][1] = now
                        found = True
                        break
                if not found:
                    self.good_list.append([state, now])

            elif en_detected_state is not None: # the engine did wrong in this detection
                if self.check_algorithm == 'last':
                    self.is_prev_good = 0
                    return
                for idx, good_item in enumerate(self.good_list):
                    if bm.bitmap_same(good_item[0], en_detected_state):
                        del self.good_list[idx]
                        break

    def _clean_state_history(self, frame_id):
        for idx in xrange(self.prev_best_engine_frame_id + 1, frame_id + 1):
            try:
                del self.state_history[idx]
            except KeyError as e:
                pass
        self.prev_best_engine_frame_id = frame_id

    def _log_bitmap(self, bitmap, engine_id, frame_id):
        np.save("log_bitmaps/%s_%d" % (engine_id, frame_id), bitmap)

    def handle(self, header, state):
        frame_id = header.get(gabriel.Protocol_client.JSON_KEY_FRAME_ID, None)
        engine_id = header.get(gabriel.Protocol_client.JSON_KEY_ENGINE_ID, None)
        engine_number = header.get(gabriel.Protocol_client.JSON_KEY_ENGINE_NUMBER, None)

        # refill tokens
        engine_info = self.image_queue_dict[engine_id]
        engine_info['tokens'][engine_number] += 1

        # filtering result (i.e. should we use it?)
        bitmap = None
        if state != "None":
            bitmap = np.array(json.loads(state))
        self._log_bitmap(bitmap, engine_id, frame_id)

        is_trust = False
        if bitmap is not None:
            if engine_id == self.best_engine:
                is_trust = True
        else:
            ## determine whether we can trust this state
            is_trust = self._trust(bitmap, engine_id, frame_id)

        ## update state history and performance table
        if engine_id == self.best_engine:
            self._update_good_list(bitmap, frame_id)
            self._clean_state_history(frame_id)

        header[gabriel.Protocol_measurement.JSON_KEY_APP_SYMBOLIC_TIME] = time.time()

        if is_trust:
            return state
        else:
            return None


def register_service(ip_addr, port, name, content):
    url = "http://%s:%d/services/%s" % (ip_addr, port, name)
    gabriel.network.http_post(url, content)


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

    # register custom service
    custom_service_info = {
        'ip': gabriel.network.get_ip(settings.net_interface),
        'port': config.MASTER_SERVER_PORT
        }
    register_service(ip_addr, port, ENGINE_ID, custom_service_info)

    # image receiving thread
    image_queue = Queue.Queue(gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    print "TOKEN SIZE OF OFFLOADING ENGINE: %d" % gabriel.Const.APP_LEVEL_TOKEN_SIZE
    video_streaming = gabriel.proxy.SensorReceiveClient((video_ip, video_port), image_queue)
    video_streaming.start()
    video_streaming.isDaemon = True

    # app proxy
    image_queue_dict = dict()
    result_queue = multiprocessing.Queue()

    master_proxy = MasterProxy(image_queue, image_queue_dict, engine_id = ENGINE_ID)
    master_proxy.start()
    master_proxy.isDaemon = True

    # data publish server
    results_queue = multiprocessing.Queue(100)
    result_queue = multiprocessing.Queue()

    p_data_server = gabriel.proxy.DataPublishServer(config.MASTER_SERVER_PORT, gabriel.proxy.DataPublishHandler, image_queue_dict, results_queue)
    p_data_server_thread = threading.Thread(target = p_data_server.serve_forever)
    p_data_server_thread.daemon = True
    p_data_server_thread.start()

    # result filter
    result_filter = ResultFilter(results_queue, result_queue, image_queue_dict, engine_id = ENGINE_ID)
    result_filter.start()
    result_filter.isDaemon = True

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
        if master_proxy is not None:
            master_proxy.terminate()
        if p_data_server is not None:
            p_data_server.terminate()
        if result_filter is not None:
            result_filter.terminate()
        if result_pub is not None:
            result_pub.terminate()

