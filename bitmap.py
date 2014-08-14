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
        row_idx, col_idx_start, col_idx_end, _, _ = diff_piece
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
    row_idx, col_idx_start, col_idx_end, _, label = bm_diff['first_piece']
    # first part of message
    if action == config.ACTION_ADD:
        message = "Now find a 1x%d %s piece and add it to " % ((col_idx_end - col_idx_start + 1), config.COLOR_ORDER[label])
        bm = bm_new
    elif action == config.ACTION_REMOVE:
        message = "This is incorrect. Now remove the 1x%d %s piece from " % ((col_idx_end - col_idx_start + 1), config.COLOR_ORDER[label])
        bm = bm_old

    is_top = row_idx == 0 or not np.any(bm[row_idx - 1, col_idx_start : col_idx_end + 1])
    is_bottom = row_idx == shape[0] - 1 or not np.any(bm[row_idx + 1, col_idx_start : col_idx_end + 1])
    is_left = col_idx_start == 0 or not np.any(bm[row_idx, 0 : col_idx_start])
    is_right = col_idx_end == shape[1] - 1 or not np.any(bm[row_idx, col_idx_end + 1 :])
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

def get_piece_direction(bm, piece):
    row_idx, col_idx_start, col_idx_end = piece
    direction = config.DIRECTION_NONE
    if row_idx == 0 or np.all(bm[row_idx - 1, col_idx_start : col_idx_end + 1] == 0):
        direction = config.DIRECTION_UP
    elif row_idx == bm.shape[0] - 1 or np.all(bm[row_idx + 1, col_idx_start : col_idx_end + 1] == 0):
        direction = config.DIRECTION_DOWN
    return direction

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
        direction = get_piece_direction(bm2, (row_idx, col_idx_start, col_idx_end))
        bm_more['first_piece'] = [row_idx, col_idx_start, col_idx_end, direction, bm_more['labels'][row_idx, col_idx_start]]
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
            bm1_large = shift_bitmap(bm1, (row_shift, col_shift), shape2)
            bm_more = bitmap_more_equalsize(bm1_large, bm2)
            if bm_more is not None:
                bm_more['shift'] = (row_shift, col_shift)
                return bm_more
    return None

def bitmap_diff(bm1, bm2):
    '''
    Detect how the two bitmaps @bm1 and @bm2 differ
    Currently can only detect the difference if one bitmap is strictly larger than the other one
    Returns @bm_diff = {
    'pieces' : an array with size equal to the bitmap showing which parts are new
    'labels' : an array with size equal to the bitmap showing what the new parts are
    'n_diff_pieces' : an integer. the number of new pieces
    'first_piece' : info about the first new piece in format [row_idx, col_idx_start, col_idx_end, direction, label]
        direction = 0 (in the middle of some pieces), 1 (on top) or 2 (on the bottom)
        This field is None if bm1 equals bm2
    'shift' : number of rows and columns the smaller bitmaps to shift to best match the big one
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

def shift_bitmap(bm, shift, final_shape):
    shape = bm.shape
    bm_shift = np.zeros(final_shape)
    bm_shift[shift[0] :  shift[0] + shape[0], shift[1] : shift[1] + shape[1]] = bm
    return bm_shift

def shrink_bitmap(bm):
    shape = bm.shape
    i_start = 0
    while i_start <= shape[0] - 1 and np.all(bm[i_start, :] == 0):
        i_start += 1
    i_end = shape[0] - 1
    while i_end >= 0 and np.all(bm[i_end, :] == 0):
        i_end -= 1
    j_start = 0
    while j_start <= shape[1] - 1 and np.all(bm[:, j_start] == 0):
        j_start += 1
    j_end = shape[1] - 1
    while j_end >= 0 and np.all(bm[:, j_end] == 0):
        j_end -= 1
    return bm[i_start : i_end + 1, j_start : j_end + 1]

def extend_piece(piece, bm):
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    # extend toward left
    j = col_idx_start - 1
    while j >= 0 and bm[row_idx, j] == label and get_piece_direction(bm, (row_idx, j, col_idx_end)) != config.DIRECTION_NONE:
        j -= 1
    col_idx_start = j + 1
    # extend toward right
    j = col_idx_end + 1
    while j <= bm.shape[1] - 1 and bm[row_idx, j] == label and get_piece_direction(bm, (row_idx, col_idx_start, j)) != config.DIRECTION_NONE:
        j += 1
    col_idx_end = j - 1
    return (row_idx, col_idx_start, col_idx_end, get_piece_direction(bm, (row_idx, col_idx_start, col_idx_end)), label)

def add_piece(bm, piece):
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    bm_ret = bm.copy()
    for j in xrange(col_idx_start, col_idx_end + 1):
        bm_ret[row_idx, j] = label
    return bm_ret

def remove_piece(bm, piece):
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    bm_ret = bm.copy()
    for j in xrange(col_idx_start, col_idx_end + 1):
        bm_ret[row_idx, j] = 0
    bm_ret = shrink_bitmap(bm_ret)
    return bm_ret

def piece_same(piece1, piece2):
    row_idx1, col_idx_start1, col_idx_end1, direction1, label1 = piece1
    row_idx2, col_idx_start2, col_idx_end2, direction2, label2 = piece2
    return col_idx_start1 - col_idx_end1 == col_idx_start2 - col_idx_end2 and label1 == label2

def shift_piece(piece, shift):
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    return (row_idx + shift[0], col_idx_start + shift[1], col_idx_end + shift[1], direction, label)

def equalize_size(bm1, bm2, common_shift1, common_shift2):
    shift1 = [0, 0]
    shift2 = [0, 0]
    if common_shift1[0] > common_shift2[0]:
        shift2[0] = common_shift1[0] - common_shift2[0]
    else:
        shift1[0] = common_shift2[0] - common_shift1[0]
    if common_shift1[1] > common_shift2[1]:
        shift2[1] = common_shift1[1] - common_shift2[1]
    else:
        shift1[1] = common_shift2[1] - common_shift1[1]
    final_shape = (max(bm1.shape[0] + shift1[0], bm2.shape[0] + shift2[0]),
                   max(bm1.shape[1] + shift1[1], bm2.shape[1] + shift2[1]))

    return (shift_bitmap(bm1, shift1, final_shape), shift_bitmap(bm2, shift2, final_shape), shift1, shift2)

