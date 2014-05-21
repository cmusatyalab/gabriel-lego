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
#import traceback
import numpy as np
#from datetime import datetime

import lego_config as config

if os.path.isdir("../../../"):
    sys.path.insert(0, "../../../")

LOG_TAG = "LEGO: "
current_milli_time = lambda: int(round(time.time() * 1000))

def raw2cv_image(raw_data):
    img_array = np.asarray(bytearray(raw_data), dtype=np.int8)
    cv_image = cv2.imdecode(img_array, -1)
    return cv_image

def display_image(display_name, img, wait_time = config.DISPLAY_WAIT_TIME, is_resize = True):
    display_max_pixel = config.DISPLAY_MAX_PIXEL
    if is_resize:
        img_shape = img.shape
        height = img_shape[0]; width = img_shape[1]
        if height > width:
            img_display = cv2.resize(img, (display_max_pixel * width / height, display_max_pixel), interpolation = cv2.INTER_NEAREST)
        else:
            img_display = cv2.resize(img, (display_max_pixel, display_max_pixel * height / width), interpolation = cv2.INTER_NEAREST)
    else:
        img_display = img
    cv2.imshow(display_name, img_display)
    cv2.waitKey(wait_time)

def super_bitwise_or(masks):
    final_mask = None
    for mask in masks:
        if final_mask is None:
            final_mask = mask
            continue
        final_mask = cv2.bitwise_or(final_mask, mask)
    return final_mask

def detect_color(img_hsv, color):
    '''
    detect the area in @img_hsv with a specific @color, and return the @mask
    @img_hsv is the input in HSV color space
    @color is a string, describing color
    Currently supported colors: Black, White, Blue, Green, Red, Yellow
    In OpenCV HSV space, H is in [0, 179], the other two are in [0, 255]
    '''
    if color == "black_dots":
        lower_bound = [0, 0, 0]
        upper_bound = [179, config.BLACK_BOARD['S_U'], config.BLACK_BOARD['B_U']]
    elif color == "black":
        lower_bound = [0, 0, 0]
        upper_bound = [179, config.BLACK['S_U'], config.BLACK['B_U']]
    elif color == "white":
        lower_bound = [0, 0, config.WHITE['B_L']]
        upper_bound = [179, config.WHITE['S_U'], 255]
    elif color == "red":
        lower_bound = [config.RED['H'] - config.HUE_RANGE, config.RED['S_L'], 0]
        upper_bound = [config.RED['H'] + config.HUE_RANGE, 255, 255]
    elif color == "green":
        lower_bound = [config.GREEN['H'] - config.HUE_RANGE, config.GREEN['S_L'], 0]
        upper_bound = [config.GREEN['H'] + config.HUE_RANGE, 255, 255]
    elif color == "blue":
        lower_bound = [config.BLUE['H'] - config.HUE_RANGE, config.BLUE['S_L'], 0]
        upper_bound = [config.BLUE['H'] + config.HUE_RANGE, 255, 255]
    elif color == "yellow":
        lower_bound = [config.YELLOW['H'] - config.HUE_RANGE, config.YELLOW['S_L'], 0]
        upper_bound = [config.YELLOW['H'] + config.HUE_RANGE, 255, 255]

    lower_bound[0] = max(lower_bound[0], 0)
    upper_bound[0] = min(upper_bound[0], 255)
    lower_range = np.array(lower_bound, dtype=np.uint8)
    upper_range = np.array(upper_bound, dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    return mask

def detect_colors(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_blue = detect_color(img_hsv, 'blue')
    mask_green = detect_color(img_hsv, 'green')
    mask_red = detect_color(img_hsv, 'red')
    mask_yellow = detect_color(img_hsv, 'yellow')
    return (mask_blue, mask_green, mask_red, mask_yellow)

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
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def angle_dist(a1, a2, angle_range = 180):
    dist1 = a2 - a1
    if dist1 > 0:
        dist2 = a2 - angle_range - a1
    else:
        dist2 = a2 + angle_range - a1
    return dist1 if abs(dist2) > abs(dist1) else dist2

def is_roughly_convex(cnt, threshold = 0.7):
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(cnt)
    return (float(cnt_area) / hull_area > threshold)

def is_line_seg_close(line1, line2):
    pt1_1 = np.array(line1[0 : 2])
    pt1_2 = np.array(line1[2 : 4])
    pt2_1 = np.array(line2[0 : 2])
    pt2_2 = np.array(line2[2 : 4])
    l1 = euc_dist(pt1_1, pt1_2)
    l2 = euc_dist(pt2_1, pt2_2)
    v1 = pt1_2 - pt1_1
    v2 = pt2_1 - pt1_1
    v3 = pt2_2 - pt1_1
    area1 = np.absolute(np.cross(v1, v2))
    area2 = np.absolute(np.cross(v1, v3))
    if max(area1, area2) < l1 * l2 / 3:
        return True
    else:
        return False

def line_interset(a, b):
    x1 = a[0]; y1 = a[1]; x2 = a[2]; y2 = a[3]
    x3 = b[0]; y3 = b[1]; x4 = b[2]; y4 = b[3]
    d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4))
    if d:
        x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d
        y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d
    else:
        x, y = (-1, -1)
    return (x, y)

def get_corner_pts(bw, perimeter, center):
    center = (center[1], center[0]) # in (x, y) format
    perimeter = int(perimeter)

    lines = cv2.HoughLinesP(bw, 1, np.pi/180, perimeter / 30, minLineLength = perimeter / 15, maxLineGap = perimeter / 15)
    lines = lines[0]
    new_lines = list()
    for line in lines:
        flag = True
        for new_line in new_lines:
            if is_line_seg_close(line, new_line):
                flag = False
                break
        if flag:
            new_lines.append(list(line))
    if len(new_lines) != 4:
        return None

    corners = list()
    for idx1, line1 in enumerate(new_lines):
        for idx2, line2 in enumerate(new_lines):
            if idx1 >= idx2:
                continue
            inter_p = line_interset(line1, line2)
            dist = euc_dist(inter_p, center)
            if dist < perimeter / 3:
                corners.append(inter_p)
    if len(corners) != 4:
        return None
    # TODO: probably still need some sanity check here to see if the corners make any sense

    dtype = [('x', float), ('y', float)]
    corners = np.array(corners, dtype = dtype)
    corners = np.sort(corners, order = 'y')
    if corners[0][0] < corners[1][0]:
        ul = corners[0]; ur = corners[1]
    else:
        ul = corners[1]; ur = corners[0]
    if corners[2][0] < corners[3][0]:
        bl = corners[2]; br = corners[3]
    else:
        bl = corners[3]; br = corners[2]

    corners = np.float32([list(ul), list(ur), list(bl), list(br)])
    return corners

def get_rotation(bw):
    lines = cv2.HoughLinesP(bw, 1, np.pi/180, 10, minLineLength = 15, maxLineGap = 10)
    lines = lines[0]
    # plotting lines, for testing only ############################
    #img = np.zeros((bw.shape[0], bw.shape[1], 3), dtype=np.uint8)
    #for line in lines:
    #    pt1 = (line[0], line[1])
    #    pt2 = (line[2], line[3])
    #    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    #cv2.namedWindow('test')
    #display_image('test', img)
    ################################################################
    degrees = np.zeros(len(lines))
    for line_idx, line in enumerate(lines):
        x_diff = line[0] - line[2]
        y_diff = line[1] - line[3]
        if x_diff == 0:
            degree = np.pi / 2 # TODO
        else:
            degree = np.arctan(float(y_diff) / x_diff)
        degrees[line_idx] = degree * 180 / np.pi
        # get an angle in (-45, 45]
        if degrees[line_idx] <= 0: 
            degrees[line_idx] += 90
        if degrees[line_idx] > 45:
            degrees[line_idx] -= 90

    # now use RANSAC like algorithm to get the consensus
    max_vote = 0
    consensus_degree = None
    for degree in degrees:
        n_vote = 0
        for degree_cmp in degrees:
            angle_diff = angle_dist(degree, degree_cmp, angle_range = 90)
            if abs(angle_diff) < 5:
                n_vote += 10 - abs(angle_diff)
        if n_vote > max_vote:
            max_vote = n_vote
            consensus_degree = degree

    best_degree = 0
    for degree_cmp in degrees:
        angle_diff = angle_dist(consensus_degree, degree_cmp, angle_range = 90)
        if abs(angle_diff) < 5:
            best_degree += angle_diff * (10 - abs(angle_diff))
    best_degree = best_degree / max_vote + consensus_degree
    if best_degree > 45:
        best_degree -= 90
    if best_degree <= -45:
        best_degree += 90

    return best_degree

def smart_crop(img):
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bi = cv2.threshold(bw, 0, 1, cv2.THRESH_BINARY)
    # TODO: has a risk that the sum here may excede uint8...
    sum_0 = bi.sum(axis = 0)
    sum_1 = bi.sum(axis = 1)
    i_start = 0; i_end = bi.shape[0] - 1; j_start = 0; j_end = bi.shape[1] - 1
    i_start_cmp_val = sum_1[int(round(config.BRICK_HEIGHT / 4.0 * 2))] * 0.6 
    while sum_1[i_start] < i_start_cmp_val:
        i_start += 1
    i_end_cmp_val = sum_1[bi.shape[0] - 1 - int(round(config.BRICK_HEIGHT / 4.0 * 2))] * 0.6 
    while sum_1[i_end] < i_end_cmp_val:
        i_end -= 1
    j_start_cmp_val = sum_0[int(round(config.BRICK_WIDTH / 4.0 * 2))] * 0.6
    while sum_0[j_start] < j_start_cmp_val:
        j_start += 1
    j_end_cmp_val = sum_0[bi.shape[1] - 1 - int(round(config.BRICK_WIDTH / 4.0 * 2))] * 0.6 
    while sum_0[j_end] < j_end_cmp_val:
        j_end -= 1
    
    #print (bi.shape, i_start, i_end, j_start, j_end)
    return img[i_start : i_end + 1, j_start : j_end + 1, :]

def mask2bool(masks):
    bools = []
    for mask in masks:
        mask[mask == 255] = 1
        mask = mask.astype(bool) 
        bools.append(mask)
    return bools

def calc_color_cumsum(img):
    height, width, _ = img.shape
    blue, green, red, yellow = detect_colors(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    black = detect_color(hsv, 'black')
    white = detect_color(hsv, 'white')
    blue, green, red, yellow, black, white = mask2bool((blue, green, red, yellow, black, white))
    nothing = np.bitwise_and(np.bitwise_and(img[:,:,0] == 0, img[:,:,1] == 0), img[:,:,2] == 0)
    black = np.bitwise_and(black, np.invert(nothing))

    nothing_cumsum = np.cumsum(np.cumsum(nothing, axis=0), axis=1)
    white_cumsum = np.cumsum(np.cumsum(white, axis=0), axis=1)
    green_cumsum = np.cumsum(np.cumsum(green, axis=0), axis=1)
    yellow_cumsum = np.cumsum(np.cumsum(yellow, axis=0), axis=1)
    red_cumsum = np.cumsum(np.cumsum(red, axis=0), axis=1)
    blue_cumsum = np.cumsum(np.cumsum(blue, axis=0), axis=1)
    black_cumsum = np.cumsum(np.cumsum(black, axis=0), axis=1)
    
    colors = {'nothing' : nothing,
              'white'   : white,
              'green'   : green,
              'yellow'  : yellow,
              'red'     : red,
              'blue'    : blue,
              'black'   : black,
             }
    color_cumsums = {'nothing' : nothing_cumsum,
                     'white'   : white_cumsum,
                     'green'   : green_cumsum,
                     'yellow'  : yellow_cumsum,
                     'red'     : red_cumsum,
                     'blue'    : blue_cumsum,
                     'black'   : black_cumsum,
                    }
    for color_key, color_cumsum in color_cumsums.iteritems():
        new_color_cumsum = np.zeros((height + 1, width + 1))
        new_color_cumsum[1:,1:] = color_cumsum
        color_cumsums[color_key] = new_color_cumsum

    return (colors, color_cumsums)


def img2bitmap(img, color_cumsums, n_rows, n_cols, lego_color):
    height, width, _ = img.shape
    img_plot = None
    bitmap = np.zeros((n_rows, n_cols))
    best_ratio = 0
    best_bitmap = None
    best_plot = None
    best_offset = None

    offset_range = {'t' : 0,
                    'b' : int(round(config.BRICK_HEIGHT / 2)),
                    'l' : int(round(config.BRICK_WIDTH / 3)),
                    'r' : int(round(config.BRICK_WIDTH / 3))}
   
    for height_offset_t in xrange(0, offset_range['t'] + 1, 2):
        for height_offset_b in xrange(0, offset_range['b'] + 1, 2):
            for width_offset_l in xrange(0, offset_range['l'] + 1, 2):
                for width_offset_r in xrange(0, offset_range['r'] + 1, 2):
                    if 'plot_line' in config.DISPLAY_LIST:
                        img_plot = lego_color.copy()

                    test_height = height - height_offset_t - height_offset_b
                    test_width = width - width_offset_l - width_offset_r
                    block_height = float(test_height) / n_rows
                    block_width = float(test_width) / n_cols
                    n_pixels = float(test_height * test_width)
                    n_good_pixels = 0
                    for i in xrange(n_rows):
                        i_start = int(round(block_height * i)) + height_offset_t + 2 # focus more on center part
                        i_end = int(round(block_height * (i + 1))) + height_offset_t - 2
                        for j in xrange(n_cols):
                            j_start = int(round(block_width * j)) + width_offset_l + 2
                            j_end = int(round(block_width * (j + 1))) + width_offset_l - 2
                            if 'plot_line' in config.DISPLAY_LIST:
                                cv2.line(img_plot, (j_end, 0), (j_end, height - 1), (0, 255, 0), 1)
                                cv2.line(img_plot, (0, i_end), (width - 1, i_end), (0, 255, 0), 1)
                            color_sum = {}
                            for color_key, color_cumsum in color_cumsums.iteritems():
                                color_sum[color_key] = color_cumsum[i_end, j_end] - color_cumsum[i_start, j_end] - color_cumsum[i_end, j_start] + color_cumsum[i_start, j_start]
                            # order: nothing, white, green, yellow, red, blue, black
                            counts = [color_sum['nothing'], color_sum['white'], color_sum['green'], color_sum['yellow'], color_sum['red'], color_sum['blue'], color_sum['black']]
                            color_idx = np.argmax(counts)
                            color_cumsum = color_cumsums[config.COLOR_ORDER[color_idx]]
                            n_good_pixels += color_cumsum[i_end+2, j_end+2] - color_cumsum[i_start-2, j_end+2] - color_cumsum[i_end+2, j_start-2] + color_cumsum[i_start-2, j_start-2]
                            bitmap[i, j] = color_idx
                    ratio = n_good_pixels / n_pixels
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_bitmap = bitmap.copy()
                        best_plot = img_plot
                        best_offset = (height_offset_t, height_offset_b, width_offset_l, width_offset_r)

    return best_bitmap, best_ratio, best_plot, best_offset

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

##################### Below are only for the Lego task #########################
def locate_lego(img, display_list):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_black = detect_color(img_hsv, 'black_dots')

    ## 1. find black dots (somewhat black, and small)
    ## 2. find area where black dots density is high
    mask_black_dots = np.zeros(mask_black.shape, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask_black, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE )
    bd_counts = np.zeros((config.BD_COUNT_N_ROW, config.BD_COUNT_N_COL)) # count black dots in each block
    for cnt_idx, cnt in enumerate(contours):
        if len(cnt) > config.BD_MAX_PERI or (hierarchy[0, cnt_idx, 3] != -1):
            continue
        max_p = cnt.max(axis = 0)
        min_p = cnt.min(axis = 0)
        diff_p = max_p - min_p
        if diff_p.max() > config.BD_MAX_SPAN:
            continue
        mean_p = cnt.mean(axis = 0)[0]
        bd_counts[int(mean_p[1] / config.BD_BLOCK_HEIGHT), int(mean_p[0] / config.BD_BLOCK_WIDTH)] += 1
        if 'black_dots' in display_list:
            cv2.drawContours(mask_black_dots, contours, cnt_idx, 255, -1)

    if 'black_dots' in display_list:
        display_image('black_dots', mask_black_dots)

    ## find a point that we are confident is in the board
    max_idx = bd_counts.argmax()
    i, j = ind2sub((config.BD_COUNT_N_ROW, config.BD_COUNT_N_COL), max_idx)
    if bd_counts[i, j] < config.BD_COUNT_THRESH:
        rtn_msg = {'status' : 'fail', 'message' : 'Too little black dots, maybe image blurred'}
        return (rtn_msg, None, None)
    in_board_p = ((i + 0.5) * config.BD_BLOCK_HEIGHT, (j + 0.5) * config.BD_BLOCK_WIDTH)

    ## locate the board by finding the contour that is likely to be of the board
    min_dist = 10000
    closest_cnt = None
    for cnt_idx, cnt in enumerate(contours):
        if hierarchy[0, cnt_idx, 3] == -1:
            continue
        max_p = cnt.max(axis = 0)
        min_p = cnt.min(axis = 0)
        #print "max: %s, min: %s" % (max_p, min_p)
        diff_p = max_p - min_p
        if diff_p.min() > config.BD_BLOCK_SPAN:
            mean_p = cnt.mean(axis = 0)[0]
            mean_p = mean_p[::-1]
            dist = euc_dist(mean_p, in_board_p)
            if dist < min_dist:
                min_dist = dist
                closest_cnt = cnt

    if min_dist > config.BOARD_MAX_DIST2P or not is_roughly_convex(closest_cnt):
        rtn_msg = {'status' : 'fail', 'message' : 'Cannot locate board border'}
        return (rtn_msg, None, None)
    hull = cv2.convexHull(closest_cnt)
    mask_board = np.zeros(mask_black.shape, dtype=np.uint8)
    cv2.drawContours(mask_board, [hull], 0, 255, -1)
    img_board = np.zeros(img.shape, dtype=np.uint8)
    img_board = cv2.bitwise_and(img, img, dst = img_board, mask = mask_board)
    if 'board' in display_list:
        display_image('board', img_board)
    
    ## some properties of the board
    board_area = cv2.contourArea(hull)
    if board_area < config.BOARD_MIN_AREA:
        rtn_msg = {'status' : 'fail', 'message' : 'Board too small'}
        return (rtn_msg, None, None)
    M = cv2.moments(hull)
    board_center = (int(M['m01']/M['m00']), int(M['m10']/M['m00'])) # in (row, col) format
    board_perimeter = cv2.arcLength(hull, True)
    #print (board_area, board_center, board_perimeter)

    ## find the perspective correction matrix
    board_border = np.zeros(mask_black.shape, dtype=np.uint8)
    cv2.drawContours(board_border, [hull], 0, 255, 1)
    corners = get_corner_pts(board_border, board_perimeter, board_center)
    if corners is None:
        rtn_msg = {'status' : 'fail', 'message' : 'Cannot locate board corners, probably because of occlusion'}
        return (rtn_msg, None, None)
    target_points = np.float32([[0, 0], [config.BOARD_RECONSTRUCT_WIDTH, 0], [0, config.BOARD_RECONSTRUCT_HEIGHT], [config.BOARD_RECONSTRUCT_WIDTH, config.BOARD_RECONSTRUCT_HEIGHT]])
    perspective_mtx = cv2.getPerspectiveTransform(corners, target_points)

    ## locate lego
    bw_board = cv2.cvtColor(img_board, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(bw_board, 50, 100, apertureSize = 3)
    if 'board_edge' in display_list:
        display_image('board_edge', edges)
    kernel = np.ones((6,6),np.int8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations = 1)
    kernel = np.ones((3,3),np.int8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations = 1)
    edges_inv = np.zeros(edges.shape, dtype=np.uint8)
    edges_inv = cv2.bitwise_not(edges, dst = edges_inv, mask = mask_board)

    mask_board_blue, mask_board_green, mask_board_red, mask_board_yellow = detect_colors(img_board)
    mask = super_bitwise_or((edges_inv, mask_board_blue, mask_board_green, mask_board_red, mask_board_yellow))
    if 'edge_inv' in display_list:
        display_image('edge_inv', mask)

    contours, hierarchy = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE )
    max_area = 0
    lego_cnt = None
    for cnt_idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < board_area / 300.0: # magic number
            continue
        if hierarchy[0, cnt_idx, 3] != -1 or not is_roughly_convex(cnt, threshold = 0.2):
            continue
        mean_p = cnt.mean(axis = 0)[0]
        mean_p = mean_p[::-1]
        if euc_dist(mean_p, board_center) > board_perimeter / 12.0: # magic number
            continue
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > max_area:
            max_area = cnt_area
            lego_cnt = cnt

    if lego_cnt is None:
        rtn_msg = {'status' : 'fail', 'message' : 'Cannot find Lego on the board'}
        return (rtn_msg, None, None)

    mask_lego = np.zeros(mask_board.shape, dtype=np.uint8)
    cv2.drawContours(mask_lego, [lego_cnt], 0, 255, -1)
    img_lego = np.zeros(img.shape, dtype=np.uint8)
    img_lego = cv2.bitwise_and(img, img, dst = img_lego, mask = mask_lego)
    # treat white brick differently to prevent it from erosion
    hsv_lego = cv2.cvtColor(img_lego, cv2.COLOR_BGR2HSV)
    mask_lego_white = detect_color(hsv_lego, 'white')
    kernel = np.uint8([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    mask_lego = cv2.erode(mask_lego, kernel, iterations = 9) # TODO: smartly select the number of erosions
    mask_lego = cv2.bitwise_or(mask_lego, mask_lego_white)
    img_lego = np.zeros(img.shape, dtype=np.uint8)
    img_lego = cv2.bitwise_and(img, img, dst = img_lego, mask = mask_lego) # this is weird, if not providing an input image, the output will be with random backgrounds... how is dst initialized?

    if 'board_corrected' in display_list:
        img_board_corrected = cv2.warpPerspective(img_board, perspective_mtx, (config.BOARD_RECONSTRUCT_WIDTH, config.BOARD_RECONSTRUCT_HEIGHT))
        display_image('board_corrected', img_board_corrected)
    if 'lego' in display_list:
        display_image('lego', img_lego)

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, img_lego, perspective_mtx)

def correct_orientation(img_lego, perspective_mtx, display_list):
    ## correct perspective
    img_perspective = cv2.warpPerspective(img_lego, perspective_mtx, (config.BOARD_RECONSTRUCT_WIDTH, config.BOARD_RECONSTRUCT_HEIGHT), flags = cv2.INTER_NEAREST)
    if 'lego_perspective' in display_list:
        display_image('lego_perspective', img_perspective)

    ## correct rotation
    img = img_perspective
    for iter in xrange(2): #Sometimes need multiple iterations to get the rotation right
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(bw, 100, 200)
        rotation_degree = get_rotation(edges)
        #print rotation_degree
        img_shape = img.shape
        M = cv2.getRotationMatrix2D((img_shape[1]/2, img_shape[0]/2), rotation_degree, scale = 1)
        img_correct = cv2.warpAffine(img, M, (img_shape[1], img_shape[0]))
        img = img_correct

    if 'lego_edge' in display_list:
        display_image('lego_edge', edges)
    if 'lego_correct' in display_list:
        display_image('lego_correct', img_correct)

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, img_correct)

def reconstruct_lego(img_lego, display_list):
    ## crop image to only the lego size
    bw_lego = cv2.cvtColor(img_lego, cv2.COLOR_BGR2GRAY)
    rows, cols = np.nonzero(bw_lego)
    min_row = min(rows); max_row = max(rows)
    min_col = min(cols); max_col = max(cols)
    img_lego_cropped = img_lego[min_row : max_row + 1, min_col : max_col + 1, :]
    img_lego_cropped = smart_crop(img_lego_cropped)
    if 'lego_cropped' in display_list:
        display_image('lego_cropped', img_lego_cropped)

    height, width, _ = img_lego_cropped.shape
    print height / config.BRICK_HEIGHT, width / config.BRICK_WIDTH
    n_rows_opt = max(int((height / config.BRICK_HEIGHT) + 0.3), 1)
    n_cols_opt = max(int((width / config.BRICK_WIDTH) + 0.5), 1)
    best_ratio = 0
    best_bitmap = None
    best_plot = None
    best_offset = None

    color_masks, color_cumsums = calc_color_cumsum(img_lego_cropped)
    np.set_printoptions(threshold=np.nan)
    print color_masks['red']
    if 'lego_color' in display_list:
        labels = np.zeros(color_masks['nothing'].shape, dtype=np.uint8) 
        labels[color_masks['black']] = 0 # not necessary
        labels[color_masks['blue']] = 1
        labels[color_masks['green']] = 2
        labels[color_masks['red']] = 3
        labels[color_masks['white']] = 4
        labels[color_masks['yellow']] = 5
        labels[color_masks['nothing']] = 6
        palette = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255],
                            [255,255,255], [0,255,255], [128,128,128]], dtype=np.uint8)
        lego_color = palette[labels]

        display_image('lego_color', lego_color)

    for n_rows in xrange(n_rows_opt - 0, n_rows_opt + 1):
        for n_cols in xrange(n_cols_opt - 0, n_cols_opt + 1):
            bitmap, ratio, img_plot, offset = img2bitmap(img_lego_cropped, color_cumsums, n_rows, n_cols, lego_color)
            print ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_bitmap = bitmap
                best_plot = img_plot
                best_offset = offset
    if 'plot_line' in display_list:
        display_image('plot_line', best_plot)

    if best_ratio < 0.8 or best_bitmap.shape != (n_rows_opt, n_cols_opt):
        rtn_msg = {'status' : 'fail', 'message' : 'Not confident about reconstruction, maybe too much noise'}
        return (rtn_msg, None)

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, best_bitmap)
