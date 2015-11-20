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

import time
import random
import bitmap as bm
import config

class Task:
    def __init__(self, bitmaps):
        self.current_state = None
        self.states = bitmaps
        self.prev_good_state = self.states[0]
        self.prev_time = None
        self.current_time = time.time()
        self.good_word_idx = 0

    def get_state(self, state_idx):
        try:
            return self.states[state_idx]
        except IndexError:
            return None

    def update_state(self, bitmap):
        self.current_state = bitmap
        self.prev_time = self.current_time
        self.current_time = time.time()

    def is_final_state(self):
        return bm.bitmap_same(self.current_state, self.get_state(-1))

    def get_first_guidance(self):
        result = {}
        target = self.get_state(0)
        result['action'] = config.ACTION_TARGET
        result['message'] = "Welcome to the Lego task. As a first step, please find a piece of 1x%d %s brick and put it on the board." % (target.shape[1], config.COLOR_ORDER[target[0, 0]])
        result['image'] = target.tolist()
        img_guidance = bm.bitmap2guidance_img(target, None, result['action'])
        return result, img_guidance

    def search_next(self, current_state, bm_diffs, search_type = 'more'):
        if bm_diffs is None:
            bm_diffs = []
            for state in self.states:
                bm_diff = bm.bitmap_diff(current_state, state)
                bm_diffs.append(bm_diff)

        next_states = []
        for idx, bm_diff in enumerate(bm_diffs):
            if bm_diff is not None and bm_diff['n_diff_pieces'] == 1:
                if search_type == 'more' and bm_diff['larger'] == 2: # exactly one more piece
                    next_states.append(self.get_state(idx))
                elif search_type == 'less' and bm_diff['larger'] == 1: # exactly one less piece
                    next_states.append(self.get_state(idx))
        return next_states, bm_diffs

    def get_guidance(self):
        result = {}

        ## Task is done
        if self.is_final_state():
            result['action'] = config.ACTION_TARGET
            result['message'] = "You have completed the task. Congratulations!"
            result['image'] = self.current_state.tolist()
            img_guidance = bm.bitmap2guidance_img(self.current_state, None, result['action'])
            return result, img_guidance

        states_more, bm_diffs = self.search_next(self.current_state, None, search_type = 'more')
        ## Case 1, next step is adding a piece
        if states_more:
            state_more = states_more[0] # for now, just pick the first possible next state
            self.prev_good_state = self.current_state
            bm_diff = bm.bitmap_diff(self.current_state, state_more)
            result['action'] = config.ACTION_ADD
            result['message'] = bm.generate_message(self.current_state, state_more, result['action'],
                                                    bm_diff['first_piece'], step_time = self.current_time - self.prev_time,
                                                    good_word_idx = self.good_word_idx)
            self.good_word_idx  = (self.good_word_idx + random.randint(1, 3)) % 4
            result['image'] = state_more.tolist()
            result['diff_piece'] = bm_diff['first_piece']
            img_guidance = bm.bitmap2guidance_img(state_more, result['diff_piece'], result['action'])
            return result, img_guidance

        states_less, _ = self.search_next(self.current_state, bm_diffs, search_type = 'less')
        ## Case 2, don't know what piece to pick next, so just deliver the target
        if not states_less:
            result['action'] = config.ACTION_TARGET
            result['message'] = "This is incorrect, please undo the last step and revert to the model shown on the screen."
            result['image'] = self.prev_good_state.tolist()
            img_guidance = bm.bitmap2guidance_img(self.prev_good_state, None, result['action'])
            return result, img_guidance

        ## Case 3, next step is moving a piece
        for state_less in states_less:
            bm_diff_less = bm.bitmap_diff(self.current_state, state_less)
            piece_less = bm.extend_piece(bm_diff_less['first_piece'], self.current_state)
            state_less_new = bm.remove_piece(self.current_state, piece_less)

            states_move_possible, _ = self.search_next(state_less_new, None, search_type = 'more')
            for state_move_possible in states_move_possible:
                bm_diff_move = bm.bitmap_diff(state_less_new, state_move_possible)
                piece_move = bm_diff_move['first_piece']
                if bm.piece_same(piece_less, piece_move):
                    state_old, state_new, shift_old, shift_new = bm.equalize_size(self.current_state, state_move_possible, bm_diff_less['shift'], bm_diff_move['shift'])
                    piece_less = bm.shift_piece(piece_less, shift_old)
                    piece_move = bm.shift_piece(piece_move, shift_new)
                    result['action'] = config.ACTION_MOVE
                    result['message'] = bm.generate_message(state_old, state_new, result['action'], piece_less, diff_piece2 = piece_move, step_time = self.current_time - self.prev_time)
                    result['image'] = bm.add_piece(state_old, piece_move).tolist()
                    result['diff_piece'] = piece_less
                    result['diff_piece2'] = piece_move
                    img_guidance = bm.bitmap2guidance_img(state_less_new, None, result['action'])
                    return result, img_guidance

        ## Case 4, next step is removing a piece
        state_less = states_less[0]
        bm_diff = bm.bitmap_diff(self.current_state, state_less)
        result['action'] = config.ACTION_REMOVE
        result['message'] = bm.generate_message(self.current_state, state_less, result['action'],
                                                bm_diff['first_piece'], step_time = self.current_time - self.prev_time)
        result['image'] = self.current_state.tolist()
        result['diff_piece'] = bm_diff['first_piece']
        img_guidance = bm.bitmap2guidance_img(self.current_state, result['diff_piece'], result['action'])

        return result, img_guidance

