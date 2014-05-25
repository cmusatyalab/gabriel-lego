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
    n_rows, n_cols = bitmap.shape
    img_syn = np.zeros((n_rows, n_cols, 3), dtype = np.uint8)
    for i in xrange(n_rows):
        for j in xrange(n_cols):
            if bitmap[i, j] == 1:
                img_syn[i, j, :] = 255
            elif bitmap[i, j] == 2:
                img_syn[i, j, 1] = 255
            elif bitmap[i, j] == 3:
                img_syn[i, j, 1:] = 255
            elif bitmap[i, j] == 4:
                img_syn[i, j, 2] = 255
            elif bitmap[i, j] == 5:
                img_syn[i, j, 0] = 255
            elif bitmap[i, j] == 0 or bitmap[i, j] == 7:
                img_syn[i, j, :] = 128
    return img_syn

def bitmap_same(bm1, bm2):
    '''
    Detect if two bitmaps @bm1 and @bm2 are exactly the same
    Return True if yes, False if no
    '''
    return np.array_equal(bm1, bm2)
