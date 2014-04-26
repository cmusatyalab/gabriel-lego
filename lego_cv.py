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

def display_image(display_name, img, wait_time = 500):
    img_display = cv2.resize(img, (640, 360))
    #img_display = img
    cv2.imshow(display_name, img_display)
    cv2.waitKey(wait_time)

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

def is_roughly_convex(cnt, threshold = 0.7):
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(cnt)
    return (float(cnt_area) / hull_area > threshold)


##################### Below are only for the Lego task #########################
def locate_lego(img):
    black_min = np.array([0, 0, 0], np.uint8)
    black_max = np.array([100, 100, 100], np.uint8)
    mask_black = cv2.inRange(img, black_min, black_max)
    img_black = cv2.bitwise_and(img, img, mask = mask_black)
    #display_image("black", mask_black)

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
        if hierarchy[0, cnt_idx, 3] == -1:
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

    hull = cv2.convexHull(closest_contour)
    #approx = cv2.approxPolyDP(hull, 100, True)
    mask_board = np.zeros(mask_black.shape, dtype=np.uint8)
    cv2.drawContours(mask_board, [hull], 0, 255, -1)

    board_area = cv2.contourArea(hull)
    M = cv2.moments(hull)
    board_center = (int(M['m01']/M['m00']), int(M['m10']/M['m00']))
    board_perimeter = cv2.arcLength(hull, True)
    #print (board_area, board_center, board_perimeter)

    if min_dist < 200:
        #cv2.drawContours(img, [hull], 0, (0, 255,0), 3)
        img_board = cv2.bitwise_and(img, img, mask = mask_board)
    bw_board = cv2.cvtColor(img_board, cv2.COLOR_BGR2GRAY)
    display_image("board", img_board)

    edges = cv2.Canny(bw_board, 100, 200)
    display_image("edge", edges)

    kernel = np.ones((6,6),np.int8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations = 1)
    kernel = np.ones((3,3),np.int8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations = 1)
    edges = cv2.bitwise_not(edges, mask = mask_board)

    contours, hierarchy = cv2.findContours(edges, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE )
    max_area = 0
    lego_cnt = None
    for cnt_idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < board_area / 300.0:
            continue
        if hierarchy[0, cnt_idx, 3] != -1 or not is_roughly_convex(cnt):
            continue
        mean_p = cnt.mean(axis = 0)[0]
        mean_p = mean_p[::-1]
        if euc_dist(mean_p, board_center) > board_perimeter / 10.0:
            continue
        if cv2.contourArea(cnt) > max_area:
            max_area = cv2.contourArea(cnt)
            lego_cnt = cnt

    mask_lego = np.zeros(mask_board.shape, dtype=np.uint8)
    cv2.drawContours(mask_lego, [lego_cnt], 0, 255, -1)
    img_lego = np.zeros(img.shape, dtype=np.uint8)
    img_lego = cv2.bitwise_and(img, img, dst = img_lego, mask = mask_lego) # this is weird, if not providing an input image, the output will be with random backgrounds... how is dst initialized?

    display_image("lego", img_lego)

    return img
