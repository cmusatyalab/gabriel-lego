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
import numpy as np

import lego_config as config

def bitmap2syn_img(bitmap):
    palette = np.array([[128,128,128], [255,255,255], [0,255,0], [0,255,255],
                        [0,0,255], [255,0,0], [0,0,0], [255,0,255]], dtype=np.uint8)
    img_syn = palette[bitmap]
    return img_syn

def bitmap2guidance_img(bitmap, diff_piece, action, max_height = 100, max_width = 100):
    img_syn = bitmap2syn_img(bitmap)
    scale1 = max_height / img_syn.shape[0]
    scale2 = max_width / img_syn.shape[1]
    scale = min(scale1, scale2)
    img_guidance = cv2.resize(img_syn, (img_syn.shape[1] * scale, img_syn.shape[0] * scale), interpolation = cv2.INTER_NEAREST)

    ## highlight the new piece(s)
    if diff_piece is not None:
        row_idx, col_idx_start, col_idx_end, _ = diff_piece
        row_idx_start = row_idx * scale
        row_idx_end = row_idx_start + scale - 1
        col_idx_start = col_idx_start * scale
        col_idx_end = (col_idx_end + 1) * scale - 1
        if action == config.ACTION_ADD:
            cv2.line(img_guidance, (col_idx_start, row_idx_start), (col_idx_start, row_idx_end), (255, 0, 255), 2)
            cv2.line(img_guidance, (col_idx_end, row_idx_start), (col_idx_end, row_idx_end), (255, 0, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_start), (col_idx_end, row_idx_start), (255, 0, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_end), (col_idx_end, row_idx_end), (255, 0, 255), 2)
        elif action == config.ACTION_REMOVE:
            cv2.line(img_guidance, (col_idx_start, row_idx_start), (col_idx_start, row_idx_end), (0, 255, 255), 2)
            cv2.line(img_guidance, (col_idx_end, row_idx_start), (col_idx_end, row_idx_end), (0, 255, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_start), (col_idx_end, row_idx_start), (0, 255, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_end), (col_idx_end, row_idx_end), (0, 255, 255), 2)

    return img_guidance

def generate_message(bm_old, bm_new, bm_diff, action):
    shape = bm_diff['pieces'].shape
    row_idx, col_idx_start, col_idx_end, _ = bm_diff['first_piece']
    color_idx = bm_diff['labels'][row_idx, col_idx_start]
    # first part of message
    if action == config.ACTION_ADD:
        message = "Now find a 1x%d %s piece and add it to " % ((col_idx_end - col_idx_start + 1), config.COLOR_ORDER[color_idx])
    elif action == config.ACTION_REMOVE:
        message = "This is incorrect. Now remove the 1x%d %s piece from " % ((col_idx_end - col_idx_start + 1), config.COLOR_ORDER[color_idx])

    is_top = row_idx == 0 or not np.any(bm_new[row_idx - 1, col_idx_start : col_idx_end + 1])
    is_bottom = row_idx == shape[0] - 1 or not np.any(bm_new[row_idx + 1, col_idx_start : col_idx_end + 1])
    is_left = col_idx_start == 0 or not np.any(bm_new[row_idx, 0 : col_idx_start])
    is_right = col_idx_end == shape[1] - 1 or not np.any(bm_new[row_idx, col_idx_end + 1 :])
    position = None
    if is_top:
        if is_left and not is_right:
            position = "top left"
        elif is_right and not is_left:
            position = "top right"
        else:
            position = "top"
    elif is_bottom:
        if is_left and not is_right:
            position = "bottom left"
        elif is_right and not is_left:
            position = "bottom right"
        else:
            position = "bottom"

    # second part of message
    if position is not None:
        message += "the %s of the current model" % position
    else:
        message += "the current model"

    return message

def bitmap_same(bm1, bm2):
    '''
    Detect if two bitmaps @bm1 and @bm2 are exactly the same
    Return True if yes, False if no
    '''
    return np.array_equal(bm1, bm2)

def bitmap_more_equalsize(bm1, bm2):
    '''
    Detect the difference of bitmaps @bm1 and @bm2 which are of the same size (and aligned right)
    Only consider the case where bm2 has more pieces than bm1
    Returns None if cannot find the more pieces
    '''
    shape = bm1.shape
    if shape != bm2.shape:
        return None
    bm_diff = np.not_equal(bm1, bm2)
    if not np.all(bm1[bm_diff] == 0): # can only be the case that bm2 has more pieces
        return None

    # initialize
    bm_more = None
    bm_more_pieces = np.zeros(shape, dtype = np.int)
    bm_more_labels = np.zeros(shape, dtype = np.int)

    # now start...
    i = 0
    j = 0
    n_diff_pieces = 0
    while i < shape[0]:
        if not bm_diff[i, j]:
            j += 1
            if j == shape[1]:
                i += 1
                j = 0
            continue
        n_diff_pieces += 1
        current_label = bm2[i, j]

        while j < shape[1] and bm2[i, j] == current_label and bm_diff[i, j]:
            bm_more_pieces[i, j] = n_diff_pieces
            bm_more_labels[i, j] = current_label
            j += 1
        if j == shape[1]:
            i += 1
            j = 0

    bm_more = {'pieces' : bm_more_pieces,
               'labels' : bm_more_labels,
               'n_diff_pieces' : n_diff_pieces}

    # some info about the first piece
    if n_diff_pieces >= 1:
        row_idxs, col_idxs = np.where(bm_more['pieces'] == 1)
        row_idx = row_idxs[0]
        col_idx_start = col_idxs.min()
        col_idx_end = col_idxs.max()
        direction = config.DIRECTION_NONE
        if row_idx == 0 or np.all(bm2[row_idx - 1, col_idx_start : col_idx_end + 1] == 0):
            direction = config.DIRECTION_UP
        elif row_idx == shape[1] - 1 or np.all(bm2[row_idx + 1, col_idx_start : col_idx_end + 1] == 0):
            direction = config.DIRECTION_DOWN
        bm_more['first_piece'] = [row_idx, col_idx_start, col_idx_end, direction]
    else:
        bm_more['first_piece'] = None
    return bm_more

def bitmap_more(bm1, bm2):
    '''
    Assuming bitmap @bm2 has more pieces than bitmap @bm1, try to detect it
    Returns None if cannot find the more pieces
    '''
    shape1 = bm1.shape
    shape2 = bm2.shape
    if shape1[0] > shape2[0] or shape1[1] > shape2[1]:
        return None

    for row_shift in xrange(shape2[0] - shape1[0] + 1):
        for col_shift in xrange(shape2[1] - shape1[1] + 1):
            bm1_large = np.zeros(shape2)
            bm1_large[row_shift : row_shift + shape1[0], col_shift : col_shift + shape1[1]] = bm1
            bm_more = bitmap_more_equalsize(bm1_large, bm2)
            if bm_more is not None:
                bm_more['row_shift'] = row_shift
                bm_more['col_shift'] = col_shift
                return bm_more
    return None

def bitmap_diff(bm1, bm2):
    '''
    Detect how the two bitmaps @bm1 and @bm2 differ
    Currently can only detect the difference if they differ with one piece of Lego
    Returns @bm_diff = {
    'pieces' : an array with size equal to the bitmap showing which parts are new
    'labels' : an array with size equal to the bitmap showing what the new parts are
    'n_diff_pieces' : an integer. the number of new pieces
    'first_piece' : info about the first new piece in format [row_idx, col_idx_start, col_idx_end, direction]
        direction = 0 (in the middle of some pieces), 1 (on top) or 2 (on the bottom)
        This field is None if bm1 equals bm2
    'row_shift' : number of rows the smaller bitmaps to shift to best match the big one
    'col_shift' : number of columns the smaller bitmaps to shift to best match the big one
    'larger' : an integer of either 1 or 2. 1 means bm2 is part of bm1, and vice versa
    '''
    # case 1: bm2 has one more piece
    bm_diff = bitmap_more(bm1, bm2)
    if bm_diff is not None:
        bm_diff['larger'] = 2
        return bm_diff

    # case 2: bm1 has one more piece
    bm_diff = bitmap_more(bm2, bm1)
    if bm_diff is not None:
        bm_diff['larger'] = 1
        return bm_diff

    return None


