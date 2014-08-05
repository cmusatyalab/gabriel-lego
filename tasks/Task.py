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

class Task:
    def __init__(self, bitmaps):
        self.state_idx = 0
        self.bitmaps = bitmaps
        self.n_states = len(bitmaps)

    def get_state(self, state_idx):
        try:
            return self.bitmaps[state_idx]
        except IndexError:
            return None

    def next_state(self):
        self.state_idx += 1
        return self.get_state(self.state_idx)

    def previous_state(self):
        self.state_idx -= 1
        return self.get_state(self.state_idx)

    def is_final_state(self):
        return self.state_idx == self.n_states - 1
