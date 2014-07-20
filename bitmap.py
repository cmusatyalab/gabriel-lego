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

import numpy as np

import lego_config as config

def bitmap2syn_img(bitmap):
    #n_rows, n_cols = bitmap.shape
    #img_syn = np.zeros((n_rows, n_cols, 3), dtype = np.uint8)
    #img_syn[bitmap == 1, :] = 255
    #img_syn[bitmap == 2, 1] = 255
    #img_syn[bitmap == 3, 1:] = 255
    #img_syn[bitmap == 4, 2] = 255
    #img_syn[bitmap == 5, 0] = 255
    #img_syn[bitmap == 0, :] = 128
    #img_syn[bitmap == 7, 0] = 255
    #img_syn[bitmap == 7, 2] = 255
    palette = np.array([[128,128,128], [255,255,255], [0,255,0], [0,255,255],
                        [0,0,255], [255,0,0], [0,0,0], [255,0,255]], dtype=np.uint8)
    img_syn = palette[bitmap]
    return img_syn

def bitmap_same(bm1, bm2):
    '''
    Detect if two bitmaps @bm1 and @bm2 are exactly the same
    Return True if yes, False if no
    '''
    return np.array_equal(bm1, bm2)

def bitmap_diff_equalsize(bm1, bm2):
    '''
    Detect the difference of bitmaps @bm1 and @bm2 which are of the same size
    '''
    shape = bm1.shape
    if shape != bm2.shape:
        return None
    bm_diff = np.not_equal(bm1, bm2)
    if not np.all(bm1[bm_diff] == 0): # can only be the case that bm2 has more pieces
        return None
    i = 0
    j = 0
    detected = None
    while i < shape[0]:
        if not bm_diff[i, j]:
            j += 1
            if j == shape[1]:
                i += 1
                j = 0
            continue
        if detected: # have detected different piece for the second time
            return None
        detected = {'label' : bm2[i, j], 'start_loc' : (i, j)}
        while j < shape[1] and bm2[i, j] == detected['label'] and bm_diff[i, j]:
            j += 1
        detected['end_loc'] = (i, j - 1)
        if j == shape[1]:
            i += 1
            j = 0

    return detected


def bitmap_more(bm1, bm2):
    '''
    Assuming bitmap @bm2 has exactly one more piece than bitmap @bm1, try to detect it
    Four cases: bm1 is at the top-left, top-right, bottom-left, bottom-right corner or bm2
    '''
    shape1 = bm1.shape
    shape2 = bm2.shape
    if shape1[0] > shape2[0] or shape1[1] > shape2[1]:
        return None

    # case 1
    bm1_large = np.zeros(shape2)
    bm1_large[:shape1[0], :shape1[1]] = bm1
    detected = bitmap_diff_equalsize(bm1_large, bm2)
    if detected is not None:
        return detected

    # case 2
    bm1_large = np.zeros(shape2)
    bm1_large[:shape1[0], shape2[1] - shape1[1]:] = bm1
    detected = bitmap_diff_equalsize(bm1_large, bm2)
    if detected is not None:
        return detected

    # case 3
    bm1_large = np.zeros(shape2)
    bm1_large[shape2[0] - shape1[0]:, :shape1[1]] = bm1
    detected = bitmap_diff_equalsize(bm1_large, bm2)
    if detected is not None:
        return detected

    # case 4
    bm1_large = np.zeros(shape2)
    bm1_large[shape2[0] - shape1[0]:, shape2[1] - shape1[1]:] = bm1
    detected = bitmap_diff_equalsize(bm1_large, bm2)
    if detected is not None:
        return detected

    return None

def bitmap_diff(bm1, bm2):
    '''
    Detect how the two bitmaps @bm1 and @bm2 differ
    Currently can only detect the difference if they differ with one piece of Lego
    '''
    # case 1: bm2 has one more piece
    detected = bitmap_more(bm1, bm2)
    if detected is not None:
        detected['larger'] = 2
        return detected

    # case 2: bm1 has one more piece
    detected = bitmap_more(bm2, bm1)
    if detected is not None:
        detected['larger'] = 1
        return detected

    return None

