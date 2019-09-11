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

import random
import sys
from base64 import b64encode

import cv2
import numpy as np

from gabriel_lego.lego_engine import config

sys.path.insert(0, "..")
import zhuocv3 as zc


def bitmap2syn_img(bitmap):
    '''
    Convert a bitmap to colored single-pixel-per-brick image
    '''
    palette = np.array(
        [[128, 128, 128], [255, 255, 255], [0, 255, 0], [0, 255, 255],
         [0, 0, 255], [255, 0, 0], [0, 0, 0], [255, 0, 255]], dtype=np.uint8)
    img_syn = palette[bitmap]
    # img_syn = cv2.resize(img_syn, (150,150), interpolation =
    # cv2.INTER_NEAREST)
    # img_syn_large = np.zeros([img_syn.shape[0] + 10, img_syn.shape[1] + 10,
    # img_syn.shape[2]])
    # img_syn_large[5:-5, 5:-5, :] = img_syn
    return img_syn


def bitmap2guidance_img(bitmap, diff_piece, action, max_height=100,
                        max_width=100):
    '''
    Generate single image guidance based on the target bitmap and operating
    piece (a piece that has been added/removed/moved)
    Marks the operating piece using coloed boxes if it's add/remove operation
    '''
    img_syn = bitmap2syn_img(bitmap)
    scale1 = max_height // img_syn.shape[0]
    scale2 = max_width // img_syn.shape[1]
    scale = min(scale1, scale2)
    img_guidance = cv2.resize(img_syn, (
        img_syn.shape[1] * scale, img_syn.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)

    ## highlight the new piece(s)
    if diff_piece is not None:
        row_idx, col_idx_start, col_idx_end, _, _ = diff_piece
        row_idx_start = row_idx * scale
        row_idx_end = row_idx_start + scale - 1
        col_idx_start = col_idx_start * scale
        col_idx_end = (col_idx_end + 1) * scale - 1
        if action == config.ACTION_ADD:
            cv2.line(img_guidance, (col_idx_start, row_idx_start),
                     (col_idx_start, row_idx_end), (255, 0, 255), 2)
            cv2.line(img_guidance, (col_idx_end, row_idx_start),
                     (col_idx_end, row_idx_end), (255, 0, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_start),
                     (col_idx_end, row_idx_start), (255, 0, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_end),
                     (col_idx_end, row_idx_end), (255, 0, 255), 2)
        elif action == config.ACTION_REMOVE:
            cv2.line(img_guidance, (col_idx_start, row_idx_start),
                     (col_idx_start, row_idx_end), (0, 255, 255), 2)
            cv2.line(img_guidance, (col_idx_end, row_idx_start),
                     (col_idx_end, row_idx_end), (0, 255, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_start),
                     (col_idx_end, row_idx_start), (0, 255, 255), 2)
            cv2.line(img_guidance, (col_idx_start, row_idx_end),
                     (col_idx_end, row_idx_end), (0, 255, 255), 2)

    return img_guidance


def bitmap2guidance_animation(bitmap, action, diff_piece=None, diff_piece2=None,
                              max_height=200, max_width=200):
    def enlarge_and_shift(img, row_idx, col_idx_start, col_idx_end, direction,
                          ratio):
        height = img.shape[0]
        width = img.shape[1]

        shift_color = img[row_idx, col_idx_start, :]
        scale1 = float(max_width) / width
        scale2 = float(max_height) / height
        scale = min(scale1, scale2)
        width_large = int(width * scale)
        height_large = int(height * scale)
        img_large = np.ones((max_height, max_width, img.shape[2]),
                            dtype=np.uint8) * 128
        img_stuff = img_large[(max_height - height_large) / 2: (
                                                                       max_height - height_large) / 2 + height_large,
                    (max_width - width_large) / 2: (
                                                           max_width -
                                                           width_large) /
                                                   2 + width_large]
        img_resized = cv2.resize(img, (width_large, height_large),
                                 interpolation=cv2.INTER_NEAREST)
        img_stuff[:, :, :] = img_resized  # this is like copyTo in c++

        if direction == config.DIRECTION_UP:
            img_stuff[int(row_idx * scale): int((row_idx + 1) * scale),
            int(col_idx_start * scale): int((col_idx_end + 1) * scale), :] = [
                128, 128, 128]
            img_stuff[
            int((row_idx - ratio) * scale): int((row_idx + 1 - ratio) * scale),
            int(col_idx_start * scale): int((col_idx_end + 1) * scale),
            :] = shift_color
        elif direction == config.DIRECTION_DOWN:
            img_stuff[int(row_idx * scale): int((row_idx + 1) * scale),
            int(col_idx_start * scale): int((col_idx_end + 1) * scale), :] = [
                128, 128, 128]
            img_stuff[
            int((row_idx + ratio) * scale): int((row_idx + 1 + ratio) * scale),
            int(col_idx_start * scale): int((col_idx_end + 1) * scale),
            :] = shift_color

        return img_large

    def encode_images(img_animation):
        img_animation_ret = []
        for cv_img, duration in img_animation:
            img_animation_ret.append(
                (b64encode(zc.cv_image2raw(cv_img)), duration))
        return img_animation_ret

    img_animation = []

    if diff_piece is not None:
        row_idx, col_idx_start, col_idx_end, direction, label = diff_piece
    if diff_piece2 is not None:
        row_idx2, col_idx_start2, col_idx_end2, direction2, label2 = diff_piece2

    if diff_piece is not None:
        height = bitmap.shape[0]
        width = bitmap.shape[1]
        if (
                row_idx == 0 or row_idx == height - 1) and direction != \
                config.DIRECTION_NONE:
            bitmap_new = np.zeros((bitmap.shape[0] + 1, bitmap.shape[1]),
                                  dtype=np.int)
            if row_idx == 0:
                bitmap_new[1:, :] = bitmap
                row_idx += 1
                diff_piece = shift_piece(diff_piece, (1, 0))
                if diff_piece2 is not None:
                    row_idx2 += 1
                    diff_piece2 = shift_piece(diff_piece2, (1, 0))
            else:
                bitmap_new[:-1, :] = bitmap
            bitmap = bitmap_new
    if diff_piece2 is not None:
        height = bitmap.shape[0]
        width = bitmap.shape[1]
        if (
                row_idx2 == 0 or row_idx2 == height - 1) and direction2 != \
                config.DIRECTION_NONE:
            bitmap_new = np.ones(
                (bitmap.shape[0] + 1, bitmap.shape[1], bitmap.shape[2]),
                dtype=np.uint8) * 128
            if row_idx2 == 0:
                bitmap_new[1:, :] = bitmap
                row_idx += 1
                diff_piece = shift_piece(diff_piece, (1, 0))
                row_idx2 += 1
                diff_piece2 = shift_piece(diff_piece2, (1, 0))
            else:
                bitmap_new[:-1, :] = bitmap
            bitmap = bitmap_new

    AUTM = 800  # animation_update_time_min
    if action == config.ACTION_ADD:
        img_show = bitmap2syn_img(bitmap)
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 1),
            AUTM))
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 0.5),
            AUTM))
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 0),
            3 * AUTM))
    elif action == config.ACTION_REMOVE:
        img_show = bitmap2syn_img(bitmap)
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 0),
            AUTM))
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 0.5),
            AUTM))
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 1),
            3 * AUTM))
    elif action == config.ACTION_MOVE:
        bitmap_tmp = bitmap.copy()
        bitmap_tmp = remove_piece(bitmap_tmp, diff_piece2, do_shrink=False)
        bitmap_tmp = add_piece(bitmap_tmp, diff_piece)
        img_show = bitmap2syn_img(bitmap_tmp)
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 0),
            AUTM))
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 0.25),
            AUTM))
        img_animation.append((
            enlarge_and_shift(img_show, row_idx, col_idx_start,
                              col_idx_end, direction, 0.5),
            AUTM))
        bitmap_tmp = bitmap.copy()
        bitmap_tmp = remove_piece(bitmap_tmp, diff_piece, do_shrink=False)
        bitmap_tmp = add_piece(bitmap_tmp, diff_piece2)
        img_show = bitmap2syn_img(bitmap_tmp)
        img_animation.append((enlarge_and_shift(img_show, row_idx2,
                                                col_idx_start2, col_idx_end2,
                                                direction2, 0.5), AUTM))
        img_animation.append((enlarge_and_shift(img_show, row_idx2,
                                                col_idx_start2, col_idx_end2,
                                                direction2, 0.25), AUTM))
        img_animation.append((enlarge_and_shift(img_show, row_idx2,
                                                col_idx_start2, col_idx_end2,
                                                direction2, 0), 3 * AUTM))
    else:
        img_show = bitmap2syn_img(bitmap)
        img_animation.append(
            (enlarge_and_shift(img_show, 0, 0, 0, 0, 0), 5 * AUTM))

    return encode_images(img_animation)


def get_piece_position(bm, piece):
    row_idx, col_idx_start, col_idx_end, _, label = piece
    is_top = row_idx == 0 or not np.any(
        bm[row_idx - 1, col_idx_start: col_idx_end + 1])
    is_bottom = row_idx == bm.shape[0] - 1 or not np.any(
        bm[row_idx + 1, col_idx_start: col_idx_end + 1])
    is_left = col_idx_start == 0 or not np.any(bm[row_idx, 0: col_idx_start])
    is_right = col_idx_end == bm.shape[1] - 1 or not np.any(
        bm[row_idx, col_idx_end + 1:])
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
    return position


def generate_message(bm_old, bm_new, action, diff_piece, diff_piece2=None,
                     step_time=0, good_word_idx=0):
    row_idx, col_idx_start, col_idx_end, _, label = diff_piece
    if action == config.ACTION_ADD:
        message = "Now find a 1x%d %s piece and add it to " % (
            (col_idx_end - col_idx_start + 1), config.COLOR_ORDER[label])
        position = get_piece_position(bm_new, diff_piece)
        if position is not None:
            message += "the %s of the current model." % position
        else:
            message += "the current model."
        p = 0.2
        if step_time > 10:  # magic number
            p = 0.8
        if random.random() < p:
            message = config.GOOD_WORDS[good_word_idx] + message
    elif action == config.ACTION_REMOVE:
        message = "Remove the 1x%d %s piece from " % ((col_idx_end -
                                                       col_idx_start + 1),
                                                      config.COLOR_ORDER[label])
        position = get_piece_position(bm_old, diff_piece)
        if position is not None:
            message += "the %s of the current model." % position
        else:
            message += "the current model."
    elif action == config.ACTION_MOVE:
        row_idx2, col_idx_start2, col_idx_end2, _, _ = diff_piece2
        if row_idx == row_idx2:
            if col_idx_start < col_idx_start2:
                if (col_idx_start2 <= col_idx_end + 1 or np.all(bm_old[row_idx,
                                                                col_idx_end +
                                                                1:
                                                                col_idx_start2] == 0)) and \
                        col_idx_start2 - col_idx_start <= 3:
                    message = "Now slightly move the 1x%d %s piece to the " \
                              "right by %d brick size." % (
                                  (col_idx_end - col_idx_start + 1),
                                  config.COLOR_ORDER[label],
                                  col_idx_start2 - col_idx_start)
                    if random.random() < 0.5:
                        message = "You are quite close. " + message
                else:
                    message = "This is incorrect. The 1x%d %s piece should be " \
                              "" \
                              "placed more to the right." % (
                                  (col_idx_end - col_idx_start + 1),
                                  config.COLOR_ORDER[label])
            else:
                if (col_idx_start <= col_idx_end2 + 1 or np.all(bm_old[row_idx,
                                                                col_idx_end2
                                                                + 1:
                                                                col_idx_start] == 0)) and \
                        col_idx_start - col_idx_start2 <= 3:
                    message = "Now slightly move the 1x%d %s piece to the " \
                              "left by %d brick size." % (
                                  (col_idx_end - col_idx_start + 1),
                                  config.COLOR_ORDER[label],
                                  col_idx_start - col_idx_start2)
                    if random.random() < 0.5:
                        message = "You are quite close. " + message
                else:
                    message = "This is incorrect. The 1x%d %s piece should be " \
                              "" \
                              "placed more to the left." % (
                                  (col_idx_end - col_idx_start + 1),
                                  config.COLOR_ORDER[label])
        else:
            message = "Now move the 1x%d %s piece " % (
                (col_idx_end - col_idx_start + 1), config.COLOR_ORDER[label])
            position = get_piece_position(bm_old, diff_piece)
            position2 = get_piece_position(bm_new, diff_piece2)
            if position is None or position2 is None:  # shouldn't happen
                message += "as shown on the screen"
            elif position[0] == position2[0]:  # remain on the top or bottom
                message += "to the %s of the current model." % position2
            else:
                message += "from the %s to the %s of the current model." % (
                    position, position2)

    return message


def get_piece_direction(bm, piece):
    row_idx, col_idx_start, col_idx_end = piece
    direction = config.DIRECTION_NONE
    if row_idx == 0 or np.all(
            bm[row_idx - 1, col_idx_start: col_idx_end + 1] == 0):
        direction = config.DIRECTION_UP
    elif row_idx == bm.shape[0] - 1 or np.all(
            bm[row_idx + 1, col_idx_start: col_idx_end + 1] == 0):
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
    Detect the difference of bitmaps @bm1 and @bm2 which are of the same size
    (and aligned right)
    Only consider the case where bm2 has more pieces than bm1
    Returns None if cannot find the more pieces
    '''
    shape = bm1.shape
    if shape != bm2.shape:
        return None
    bm_diff = np.not_equal(bm1, bm2)
    if not np.all(
            bm1[bm_diff] == 0):  # can only be the case that bm2 has more pieces
        return None

    # initialize
    bm_more = None
    bm_more_pieces = np.zeros(shape, dtype=np.int)
    bm_more_labels = np.zeros(shape, dtype=np.int)

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

    bm_more = {'pieces'       : bm_more_pieces,
               'labels'       : bm_more_labels,
               'n_diff_pieces': n_diff_pieces}

    # some info about the first piece
    if n_diff_pieces >= 1:
        row_idxs, col_idxs = np.where(bm_more['pieces'] == 1)
        row_idx = row_idxs[0]
        col_idx_start = col_idxs.min()
        col_idx_end = col_idxs.max()
        direction = get_piece_direction(bm2,
                                        (row_idx, col_idx_start, col_idx_end))
        bm_more['first_piece'] = [row_idx, col_idx_start, col_idx_end,
                                  direction,
                                  bm_more['labels'][row_idx, col_idx_start]]
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

    for row_shift in range(shape2[0] - shape1[0] + 1):
        for col_shift in range(shape2[1] - shape1[1] + 1):
            bm1_large = shift_bitmap(bm1, (row_shift, col_shift), shape2)
            bm_more = bitmap_more_equalsize(bm1_large, bm2)
            if bm_more is not None:
                bm_more['shift'] = (row_shift, col_shift)
                return bm_more
    return None


def bitmap_diff(bm1, bm2):
    '''
    Detect how the two bitmaps @bm1 and @bm2 differ
    Currently can only detect the difference if one bitmap is strictly larger
    than the other one
    Returns @bm_diff = {
    'pieces' : an array with size equal to the bitmap showing which parts are
    new
    'labels' : an array with size equal to the bitmap showing what the new
    parts are
    'n_diff_pieces' : an integer. the number of new pieces
    'first_piece' : info about the first new piece in format [row_idx,
    col_idx_start, col_idx_end, direction, label]
        direction = 0 (in the middle of some pieces), 1 (on top) or 2 (on the
        bottom)
        This field is None if bm1 equals bm2
    'shift' : number of rows and columns the smaller bitmaps to shift to best
    match the big one
    'larger' : an integer of either 1 or 2. 1 means bm2 is part of bm1,
    and vice versa
    '''

    # if arrays are the same return None
    if np.array_equal(bm1, bm2):
        return None

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
    bm_shift = np.zeros(final_shape, dtype=np.int)
    bm_shift[shift[0]:  shift[0] + shape[0], shift[1]: shift[1] + shape[1]] = bm
    return bm_shift


def shrink_bitmap(bm):
    '''
    Remove the all zero lines at the four sides of the bitmap
    '''
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
    return bm[i_start: i_end + 1, j_start: j_end + 1]


def extend_piece(piece, bm):
    '''
    Given a piece and a bitmap, find if the piece can be part of a longer piece.
    '''
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    # extend toward left
    j = col_idx_start - 1
    while j >= 0 and bm[row_idx, j] == label and get_piece_direction(bm, (
            row_idx, j, col_idx_end)) != config.DIRECTION_NONE:
        j -= 1
    col_idx_start = j + 1
    # extend toward right
    j = col_idx_end + 1
    while j <= bm.shape[1] - 1 and bm[
        row_idx, j] == label and get_piece_direction(bm, (
            row_idx, col_idx_start, j)) != config.DIRECTION_NONE:
        j += 1
    col_idx_end = j - 1
    return (row_idx, col_idx_start, col_idx_end,
            get_piece_direction(bm, (row_idx, col_idx_start, col_idx_end)),
            label)


def add_piece(bm, piece):
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    bm_ret = bm.copy()
    for j in range(col_idx_start, col_idx_end + 1):
        bm_ret[row_idx, j] = label
    return bm_ret


def remove_piece(bm, piece, do_shrink=True):
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    bm_ret = bm.copy()
    for j in range(col_idx_start, col_idx_end + 1):
        bm_ret[row_idx, j] = 0
    if do_shrink:
        bm_ret = shrink_bitmap(bm_ret)
    return bm_ret


def piece_same(piece1, piece2):
    row_idx1, col_idx_start1, col_idx_end1, direction1, label1 = piece1
    row_idx2, col_idx_start2, col_idx_end2, direction2, label2 = piece2
    return col_idx_start1 - col_idx_end1 == col_idx_start2 - col_idx_end2 and\
           label1 == label2


def shift_piece(piece, shift):
    row_idx, col_idx_start, col_idx_end, direction, label = piece
    return (
        row_idx + shift[0], col_idx_start + shift[1], col_idx_end + shift[1],
        direction, label)


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

    return (shift_bitmap(bm1, shift1, final_shape),
            shift_bitmap(bm2, shift2, final_shape), shift1, shift2)
