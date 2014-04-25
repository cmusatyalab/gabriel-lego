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

import os
import sys
import cv2
import time
import struct
import traceback
import numpy as np
if os.path.isdir("../../../"):
    sys.path.insert(0, "../../../")

from gabriel.proxy.common import LOG

LOG_TAG = "LEGO: "

def raw2cv_image(raw_data):
    img_array = np.asarray(bytearray(raw_data), dtype=np.int8)
    cv_image = cv2.imdecode(img_array, -1)
    return cv_image

def display_image(display_name, img):
    img_display = cv2.resize(img, (640, 360))
    #img_display = img
    cv2.imshow(display_name, img_display)
    cv2.waitKey(1)

def set_value(img, pts, value):
    '''
    set the points (@pts) in the image (@img) to value (@value)
    @img is the input image array, can be single/multi channel
    @pts are n * 2 arrays where n is the number of points
    '''
    if pts.ndim == 3:
        pts.resize(len(pts), 2)
    is_multichannel = img.ndim > 2
    i = pts[:, 1]
    j = pts[:, 0]
    if is_multichannel:
        img[i, j, :] = value
    else:
        img[i, j] = value

def ind2sub(size, idx):
    return (idx / size[1], idx % size[1])

def euc_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


##################### Below are only for the Lego task #########################
def locate_lego(img):
    black_min = np.array([0, 0, 0], np.uint8)
    black_max = np.array([100, 100, 100], np.uint8)
    mask_black = cv2.inRange(img, black_min, black_max)
    img_black = cv2.bitwise_and(img, img, mask = mask_black)
    display_image("black", mask_black)

    mask_black_copy = mask_black.copy()
    contours, hierarchy = cv2.findContours(mask_black_copy, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE )
    counts = np.zeros((9, 16))
    for cnt_idx, contour in enumerate(contours):
        if len(contour) > 20 or (hierarchy[0, cnt_idx, 3] != -1):
            continue
        max_p = contour.max(axis = 0)
        min_p = contour.min(axis = 0)
        diff_p = max_p - min_p
        if diff_p.max() > 5:
            continue
        mean_p = contour.mean(axis = 0)[0]
        counts[int(mean_p[1] / 80), int(mean_p[0] / 80)] += 1

    # find a point that we are confident is in the board
    max_idx = counts.argmax()
    i, j = ind2sub((9, 16), max_idx)
    if counts[i, j] < 50:
        display_image("board", img)
        return img

    in_board_p = (i * 80 + 40, j * 80 + 40)
    #print counts
    #print in_board_p

    # find the contours that is likely to be of the board
    min_dist = 10000
    closest_contour = None
    for cnt_idx, contour in enumerate(contours):
        if hierarchy[0, cnt_idx, 3] != -1:
            continue
        max_p = contour.max(axis = 0)
        min_p = contour.min(axis = 0)
        #print "max: %s, min: %s" % (max_p, min_p)
        diff_p = max_p - min_p
        if diff_p.min() > 100:
            mean_p = contour.mean(axis = 0)[0]
            mean_p = mean_p[::-1]
            dist = euc_dist(mean_p, in_board_p)
            if dist < min_dist:
                min_dist = dist
                closest_contour = contour

    #    cv2.floodFill(mask_black, None, tuple(contour[0, 0]), 0)

    #cv2.drawContours(img, contours, -1, (255,0,0), 3)
    if min_dist < 200:
        cv2.drawContours(img, [closest_contour], 0, (0, 255,0), 3)
    display_image("board", img)

    return img

