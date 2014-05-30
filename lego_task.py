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

import sys
import cv2
import numpy as np
import bitmap as bm
import lego_cv as lc
import lego_config as config
sys.path.insert(0, "tasks")


class Task:
    def __init__(self, task_name):
        task = __import__(task_name)
        self.bitmaps = task.bitmaps
        self.n_states = len(self.bitmaps)
        # states are bitmaps
        self.current_state = None
        self.target_state_idx = 0
        self.target_state = self.bitmaps[0]

    def get_command():
        pass

    def next_state(self):
        if self.target_state_idx == self.n_states - 1:
            return None
        self.target_state_idx += 1
        self.target_state = self.bitmaps[self.target_state_idx]
        return self.target_state

    def previous_state(self):
        pass

if __name__ == "__main__":
    test_task = Task('task_Turtle')
    bitmap = test_task.next_state()
    cv2.namedWindow('test')
    while bitmap is not None:
        img_syn = bm.bitmap2syn_img(bitmap)
        lc.display_image('test', img_syn, wait_time = 2000)
        bitmap = test_task.next_state()

