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

import bitmap as bm
import lego_config as config

class Task:
    def __init__(self, bitmaps):
        self.current_state = None
        self.states = bitmaps
        self.prev_good_state = self.states[0]

    def get_state(self, state_idx):
        try:
            return self.states[state_idx]
        except IndexError:
            return None

    def update_state(self, bitmap):
        self.current_state = bitmap

    def is_final_state(self):
        return bm.bitmap_same(self.current_state, self.states[-1])

    def get_first_guidance(self):
        result = {}
        target = self.states[0]
        result['action'] = config.ACTION_TARGET
        result['message'] = "Welcome to the Lego task. As a first step, please find a piece of 1x%d %s brick and put it on the board." % (target.shape[1], config.COLOR_ORDER[target[0, 0]]),
        result['image'] = target.tolist()
        img_guidance = bm.bitmap2guidance_img(target, None, result['action'])
        return result, img_guidance

    def get_guidance(self):
        result = {}
        
        if self.is_final_state():
            result['action'] = config.ACTION_TARGET
            result['message'] = "You have completed the task. Congratulations!"
            result['image'] = self.current_state.tolist()
            img_guidance = bm.bitmap2guidance_img(self.current_state, None, result['action'])
            return result, img_guidance

        state_more = None
        state_less = None
        for state in self.states:
            bm_diff = bm.bitmap_diff(self.current_state, state)
            if bm_diff is not None and bm_diff['n_diff_pieces'] == 1:
                if bm_diff['larger'] == 2: # exactly one more piece
                    state_more = state
                    break
                else: # exactly one less piece
                    state_less = state

        if state_more is not None:
            self.prev_good_state = self.current_state
            bm_diff = bm.bitmap_diff(self.current_state, state_more)
            result['action'] = config.ACTION_ADD
            result['message'] = bm.generate_message(self.current_state, state, bm_diff, result['action'])
            result['image'] = state.tolist()
            result['diff_piece'] = bm_diff['first_piece']
            img_guidance = bm.bitmap2guidance_img(state, result['diff_piece'], result['action'])
        elif state_less is not None:
            bm_diff = bm.bitmap_diff(self.current_state, state_less)
            result['action'] = config.ACTION_REMOVE
            result['message'] = bm.generate_message(self.current_state, state, bm_diff, result['action'])
            result['image'] = self.current_state.tolist()
            result['diff_piece'] = bm_diff['first_piece']
            img_guidance = bm.bitmap2guidance_img(self.current_state, result['diff_piece'], result['action'])
        else:
            result['action'] = config.ACTION_TARGET
            result['message'] = "This is incorrect, please undo the last step and revert to the model shown on the screen."
            result['image'] = self.prev_good_state.tolist()
            img_guidance = bm.bitmap2guidance_img(self.prev_good_state, None, result['action'])

        return result, img_guidance

