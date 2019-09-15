#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#   Extended and ported to Python 3 by: Manuel Olguín Muñoz <molguin@kth.se>
#
#   Copyright (C) 2011-2019 Carnegie Mellon University
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

from __future__ import annotations

import time
from enum import Enum
from typing import NamedTuple

import cv2
import numpy as np

from gabriel_lego.cv import bitmap as bm, zhuocv3 as zc
from gabriel_lego.lego_engine import config

LOG_TAG = "LEGO: "
current_milli_time = lambda: int(round(time.time() * 1000))


# Color mappings
# nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6, unsure:7

class HSV_Tuple(NamedTuple):
    hue: int
    saturation: int
    value: int


class LEGOCVColor:
    def __init__(self,
                 value_mapping: int,
                 range_low: HSV_Tuple,
                 range_high: HSV_Tuple):
        self.value_mapping = value_mapping
        self.lower_range = np.array([*range_low], dtype=np.uint8)
        self.upper_range = np.array([*range_high], dtype=np.uint8)

    @property
    def mapping(self):
        return self.value_mapping

    @property
    def lower_range(self):
        return self.lower_range

    @property
    def upper_range(self):
        return self.upper_range

    def __eq__(self, other: LEGOCVColor):
        return self.mapping == other.mapping


class LEGO_COLORS(LEGOCVColor, Enum):
    WHITE = LEGOCVColor(1)
    GREEN = LEGOCVColor(2)
    YELLOW = LEGOCVColor(3)
    RED = LEGOCVColor(4)
    BLUE = LEGOCVColor(5)
    BLACK = LEGOCVColor(6)

    # TODO: use them in the code


# Errors
class LEGOCVError(Exception):
    pass


class NoBoardDetectedError(LEGOCVError):
    pass


class NoLEGODetectedError(LEGOCVError):
    pass


################################ BASICS ########################################
def set_config(is_streaming):
    config.setup(is_streaming)


def has_a_brick(mask, min_peri=None, min_area=None, min_span=None,
                print_max_area=False):
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                           method=cv2.CHAIN_APPROX_NONE)
    max_area = 0
    ret = False
    for cnt_idx, cnt in enumerate(contours):
        if hierarchy[0, cnt_idx, 3] != -1:  # not holes
            continue
        if print_max_area and cv2.contourArea(cnt) > max_area:
            max_area = cv2.contourArea(cnt)
        if min_peri is not None and len(cnt) < min_peri:
            continue
        if min_area is not None and cv2.contourArea(cnt) < min_area:
            continue
        if min_span is not None:
            max_p = cnt.max(axis=0)
            min_p = cnt.min(axis=0)
            diff_p = max_p - min_p
            if diff_p.min() + 1 < min_span:
                continue
        ret = True
        break
    if print_max_area:
        print(max_area)
    return ret


################################ SHAPE #########################################
def is_line_seg_close(line1, line2):
    pt1_1 = np.array(line1[0: 2])
    pt1_2 = np.array(line1[2: 4])
    pt2_1 = np.array(line2[0: 2])
    pt2_2 = np.array(line2[2: 4])
    l1 = zc.euc_dist(pt1_1, pt1_2)
    l2 = zc.euc_dist(pt2_1, pt2_2)
    v1 = pt1_2 - pt1_1
    v2 = pt2_1 - pt1_1
    v3 = pt2_2 - pt1_1
    area1 = np.absolute(np.cross(v1, v2))
    area2 = np.absolute(np.cross(v1, v3))
    if max(area1, area2) < l1 * l2 / 3:
        return True
    else:
        return False


def is_line_seg_close2(line1, line2):
    pt1_1 = np.array(line1[0: 2])
    pt1_2 = np.array(line1[2: 4])
    pt2_1 = np.array(line2[0: 2])
    pt2_2 = np.array(line2[2: 4])
    l1 = zc.euc_dist(pt1_1, pt1_2)
    v1 = pt1_2 - pt1_1
    v2 = pt2_1 - pt1_1
    v3 = pt2_2 - pt1_1
    area1 = np.absolute(np.cross(v1, v2))
    area2 = np.absolute(np.cross(v1, v3))
    d1 = area1 * 2 / l1
    d2 = area2 * 2 / l1
    return (d1 <= 3 and d2 <= 3)


def line_interset(a, b):
    x1 = a[0];
    y1 = a[1];
    x2 = a[2];
    y2 = a[3]
    x3 = b[0];
    y3 = b[1];
    x4 = b[2];
    y4 = b[3]
    d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if d:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (
                x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (
                x3 * y4 - y3 * x4)) / d
    else:
        x, y = (-1, -1)
    return (x, y)


def get_corner_pts(bw, perimeter=None, center=None, method='line'):
    '''
    Given an input image @bw where the borders of a rough rectangle are
    masked, the function detects its corners
    Two methods:
    'line' tries to detect four lines first, and
    'point' directly gets the top-left, top-right, bottom-left, bottom-right
    points
    The function returns None if cannot find the corners with confidence
    '''
    if method == 'line':
        center = (center[1], center[0])  # in (x, y) format
        perimeter = int(perimeter)

        lines = cv2.HoughLinesP(bw, 1, np.pi / 180, perimeter // 40,
                                minLineLength=perimeter // 20,
                                maxLineGap=perimeter // 20)
        # lines = lines[0]
        # HoughLines used to have a different return format which required us
        # to extract the first element of a tuple to get the list of lines
        # the new format returns a list of 2d-arrays so instead we iterate
        # directly over the list and extract the first element of each element

        # This is only for test
        # img = np.zeros((bw.shape[0], bw.shape[1], 3), dtype=np.uint8)
        # for line in lines:
        #    pt1 = (line[0], line[1])
        #    pt2 = (line[2], line[3])
        #    print (pt1, pt2)
        #    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
        # cv2.namedWindow('test')
        # display_image('test', img, wait_time = config.DISPLAY_WAIT_TIME,
        # resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)

        # get four major lines
        new_lines = list()
        for line in lines:
            line = line[0]
            flag = True
            for new_line in new_lines:
                if is_line_seg_close(line, new_line):
                    flag = False
                    break
            if flag:
                new_lines.append(list(line))
        if len(new_lines) != 4:
            print(
                f'''
                lines: {lines}
                new_lines: {new_lines}
                '''
            )
            return None

        # get four reasonable line intersections
        corners = list()
        for idx1, line1 in enumerate(new_lines):
            for idx2, line2 in enumerate(new_lines):
                if idx1 >= idx2:
                    continue
                inter_p = line_interset(line1, line2)
                if inter_p == (-1, -1):
                    continue
                dist = zc.euc_dist(inter_p, center)
                if dist < perimeter // 3:
                    corners.append(inter_p)
        if len(corners) != 4:
            return None

        # put the four corners in order
        dtype = [('x', float), ('y', float)]
        corners = np.array(corners, dtype=dtype)
        corners = np.sort(corners, order='y')
        if corners[0][0] < corners[1][0]:
            ul = corners[0];
            ur = corners[1]
        else:
            ul = corners[1];
            ur = corners[0]
        if corners[2][0] < corners[3][0]:
            bl = corners[2];
            br = corners[3]
        else:
            bl = corners[3];
            br = corners[2]
        ul = list(ul)
        ur = list(ur)
        bl = list(bl)
        br = list(br)

        # some sanity check here
        len_b = zc.euc_dist(bl, br)
        len_u = zc.euc_dist(ul, ur)
        len_l = zc.euc_dist(ul, bl)
        len_r = zc.euc_dist(ur, br)
        if len_b < len_u or len_b < len_l or len_b < len_r:
            return None

    elif method == 'point':
        bw = bw.astype(bool)
        row_mtx, col_mtx = np.mgrid[0: bw.shape[0], 0: bw.shape[1]]
        row_mtx = row_mtx[bw]
        col_mtx = col_mtx[bw]

        row_plus_col = row_mtx + col_mtx
        ul_idx = np.argmin(row_plus_col)
        ul = (col_mtx[ul_idx], row_mtx[ul_idx])
        br_idx = np.argmax(row_plus_col)
        br = (col_mtx[br_idx], row_mtx[br_idx])

        row_minus_col = row_mtx - col_mtx
        ur_idx = np.argmin(row_minus_col)
        ur = (col_mtx[ur_idx], row_mtx[ur_idx])
        bl_idx = np.argmax(row_minus_col)
        bl = (col_mtx[bl_idx], row_mtx[bl_idx])

    corners = np.float32([ul, ur, bl, br])
    return corners


def calc_thickness(corners, stretch_ratio):
    corners_tmp = corners.copy()
    ul = corners_tmp[0]
    ul[1] *= stretch_ratio
    ur = corners_tmp[1]
    ur[1] *= stretch_ratio
    bl = corners_tmp[2]
    bl[1] *= stretch_ratio
    br = corners_tmp[3]
    br[1] *= stretch_ratio
    len_b = zc.euc_dist(bl, br)
    um = (ul + ur) / 2
    seen_board_height = zc.calc_triangle_area(bl, br, um) * 2 / len_b
    real_board_height = len_b * config.BOARD_RECONSTRUCT_HEIGHT / \
                        config.BOARD_RECONSTRUCT_WIDTH
    real_brick_height = real_board_height / config.BOARD_RECONSTRUCT_HEIGHT * \
                        config.BRICK_HEIGHT
    seen_brick_height = seen_board_height / config.BOARD_RECONSTRUCT_HEIGHT * \
                        config.BRICK_HEIGHT
    S_theta = seen_brick_height / real_brick_height  # sin theta
    if S_theta >= 1:
        C_theta = 0
    else:
        C_theta = (1 - S_theta * S_theta) ** 0.5
    real_brick_thickness = real_brick_height / \
                           config.BRICK_HEIGHT_THICKNESS_RATIO
    seen_brick_thickness = real_brick_thickness * C_theta
    seen_brick_thickness /= stretch_ratio
    return seen_brick_thickness


def get_rotation_degree(bw):
    lines = cv2.HoughLinesP(bw, 1, np.pi / 180, 6, minLineLength=8,
                            maxLineGap=5)
    if lines is None:
        return None
    lines = lines[0]
    if len(lines) > 60:
        return None
    # plotting lines, for testing only ############################
    # img = np.zeros((bw.shape[0], bw.shape[1], 3), dtype=np.uint8)
    # for line in lines:
    #    pt1 = (line[0], line[1])
    #    pt2 = (line[2], line[3])
    #    cv2.line(img, pt1, pt2, (255, 255, 255), 1)
    # cv2.namedWindow('bw')
    # display_image('bw', bw)
    # cv2.namedWindow('test')
    # display_image('test', img)
    ################################################################
    degrees = np.zeros(len(lines))
    for line_idx, line in enumerate(lines):
        x_diff = line[0] - line[2]
        y_diff = line[1] - line[3]
        if x_diff == 0:
            degree = np.pi / 2  # TODO
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
            angle_diff = zc.angle_dist(degree, degree_cmp, angle_range=90)
            if abs(angle_diff) < 5:
                n_vote += 10 - abs(angle_diff)
        if n_vote > max_vote:
            max_vote = n_vote
            consensus_degree = degree

    best_degree = 0
    for degree_cmp in degrees:
        angle_diff = zc.angle_dist(consensus_degree, degree_cmp, angle_range=90)
        if abs(angle_diff) < 5:
            best_degree += angle_diff * (10 - abs(angle_diff))
    best_degree = best_degree / max_vote + consensus_degree
    if best_degree > 45:
        best_degree -= 90
    if best_degree <= -45:
        best_degree += 90

    return best_degree


def rotate(img, n_iterations=2):
    '''
    Assuming major line patterns in an image are vertical and horizontal,
    this function tries to
    correct the rotaion to make vertical lines really vertical and horizontal
    lines really horizontal.
    '''
    img_ret = img
    rotation_degree = 0
    rotation_mtx = None
    for iteration in range(
            n_iterations):  # Sometimes need multiple iterations to get the
        # rotation right
        bw = cv2.cvtColor(img_ret, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(bw, 50, 100)
        rotation_degree_tmp = get_rotation_degree(edges)
        if rotation_degree_tmp is None:
            # rtn_msg = {'status' : 'fail',
            #            'message': 'Cannot get rotation degree'}
            # return (rtn_msg, None)
            raise LEGOCVError('Cannot get rotation degree.')
        weight = 1
        for i in range(3):
            bw[:] = img_ret[:, :, i][:]
            edges = cv2.Canny(bw, 50, 100)
            d = get_rotation_degree(edges)
            if d is not None:
                rotation_degree_tmp += d
                weight += 1
        rotation_degree_tmp /= weight
        rotation_degree += rotation_degree_tmp
        # print rotation_degree
        img_shape = img.shape
        M = cv2.getRotationMatrix2D((img_shape[1] / 2, img_shape[0] / 2),
                                    rotation_degree, scale=1)
        rotation_mtx = M
        img_ret = cv2.warpAffine(img, M, (img_shape[1], img_shape[0]))

    return img_ret, rotation_degree, rotation_mtx


def crop(img, borders):
    shape = img.shape
    is_color = (len(shape) == 3 and shape[2] > 1)
    if borders is None:
        if is_color:
            bw = zc.get_mask(img)
        else:
            bw = img
        rows, cols = np.nonzero(bw)
        min_row = min(rows);
        max_row = max(rows)
        min_col = min(cols);
        max_col = max(cols)
    else:
        min_row, max_row, min_col, max_col = borders
    if is_color:
        img_cropped = img[min_row: max_row + 1, min_col: max_col + 1, :]
    else:
        img_cropped = img[min_row: max_row + 1, min_col: max_col + 1]
    return img_cropped, (min_row, max_row, min_col, max_col)


def smart_crop(img):
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bi = cv2.threshold(bw, 0, 1, cv2.THRESH_BINARY)
    # TODO: has a risk that the sum here may excede uint8...
    sum_0 = bi.sum(axis=0)
    sum_1 = bi.sum(axis=1)
    i_start = 0;
    i_end = bi.shape[0] - 1;
    j_start = 0;
    j_end = bi.shape[1] - 1
    i_start_cmp_val = sum_1[int(round(config.BRICK_HEIGHT / 4.0 * 2))] * 0.6
    while sum_1[i_start] < i_start_cmp_val:
        i_start += 1
    i_end_cmp_val = sum_1[bi.shape[0] - 1 - int(
        round(config.BRICK_HEIGHT / 4.0 * 2))] * 0.6
    while sum_1[i_end] < i_end_cmp_val:
        i_end -= 1
    j_start_cmp_val = sum_0[int(round(config.BRICK_WIDTH / 4.0 * 2))] * 0.6
    while sum_0[j_start] < j_start_cmp_val:
        j_start += 1
    j_end_cmp_val = sum_0[bi.shape[1] - 1 - int(
        round(config.BRICK_WIDTH / 4.0 * 2))] * 0.6
    while sum_0[j_end] < j_end_cmp_val:
        j_end -= 1

    # print (bi.shape, i_start, i_end, j_start, j_end)
    return img[i_start: i_end + 1, j_start: j_end + 1, :], (
        i_start, i_end, j_start, j_end)


################################ COLOR #########################################
def normalize_brightness(img, mask=None, method='hist', max_percentile=100,
                         min_percentile=0):
    shape = img.shape
    if mask is None:
        mask = np.ones((shape[0], shape[1]), dtype=bool)
    if mask.dtype != np.bool:
        mask = mask.astype(bool)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    if method == 'hist':
        hist, bins = np.histogram(v.flatten(), 256, [0, 256])
        hist[0] = 0
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        v_ret = cdf[v]

    elif method == 'max':
        max_v = np.percentile(v[mask], max_percentile)
        min_v = np.percentile(v[mask], min_percentile)
        v[np.bitwise_and((v < min_v), mask)] = min_v
        # What the hell is converScaleAbs doing??? why need abs???
        v_ret = cv2.convertScaleAbs(v, alpha=254.0 / (max_v - min_v),
                                    beta=-(min_v * 254.0 / (max_v - min_v) - 2))
        v[mask] = v_ret[mask]
        v_ret = v

    hsv[:, :, 2] = v_ret
    img_ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img_ret


def normalize_color(img, mask_info=None, mask_apply=None, method='hist',
                    max_percentile=100, min_percentile=0):
    shape = img.shape
    if mask_info is None:
        mask_info = np.ones((shape[0], shape[1]), dtype=bool)
    if mask_info.dtype != np.bool:
        mask_info = mask_info.astype(bool)
    if mask_apply is None:
        mask_apply = mask_info
    if mask_apply.dtype != np.bool:
        mask_apply = mask_apply.astype(bool)
    img_ret = img.copy()
    if method == 'hist':  # doesn't work well for over-exposed images
        for i in range(3):
            v = img[:, :, i]
            hist, bins = np.histogram(v[mask_info].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            v[mask_apply] = cdf[v[mask_apply]]
            img_ret[:, :, i] = v

    elif method == 'grey':
        img = img.astype(float)
        max_rgb = 0
        for i in range(3):
            v = img[:, :, i]
            # print v[mask_info].mean()
            v[mask_apply] = v[mask_apply] / v[mask_info].mean()
            img[:, :, i] = v
            if v[mask_apply].max() > max_rgb:
                max_rgb = v[mask_apply].max()

        img[mask_apply, :] = img[mask_apply, :] * 255 / max_rgb
        img = img.astype(np.uint8)
        img_ret = img

    elif method == 'select_grey':
        img = img.astype(np.int64)
        mask_blue_over_exposed = (img[:, :, 0] >= 250)
        mask_green_over_exposed = (img[:, :, 1] >= 250)
        mask_red_over_exposed = (img[:, :, 2] >= 250)
        # print "Blue over exposure: %d" % mask_blue_over_exposed.sum()
        mask_over_bright = ((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) >= 666)
        mask_over_exposed = np.bitwise_and(zc.super_bitwise_or((
            mask_blue_over_exposed,
            mask_green_over_exposed,
            mask_red_over_exposed)),
            mask_over_bright)
        # print "Over exposure: %d" % mask_over_bright.sum()
        mask_info = np.bitwise_and(mask_info, np.invert(mask_over_exposed))

        img = img.astype(float)
        max_rgb = 0
        for i in range(3):
            v = img[:, :, i]
            v[mask_apply] = v[mask_apply] / v[mask_info].mean()
            img[:, :, i] = v
            if v[mask_apply].max() > max_rgb:
                max_rgb = v[mask_apply].max()

        img[mask_apply, :] = img[mask_apply, :] * 255 / max_rgb
        img = img.astype(np.uint8)
        img = normalize_brightness(img, mask=mask_apply, max_percentile=90,
                                   method='max')
        img[mask_over_exposed, 0] = 255
        img[mask_over_exposed, 1] = 255
        img[mask_over_exposed, 2] = 255
        img_ret = img

    elif method == 'max':
        # b, g, r = cv2.split(img)
        # img = cv2.merge((b, g, r))
        for i in range(3):
            v = img[:, :, i]
            max_v = np.percentile(v[mask], max_percentile)
            min_v = np.percentile(v[mask], min_percentile)
            v[v < min_v] = min_v
            v_ret = cv2.convertScaleAbs(v, alpha=220.0 / (max_v - min_v),
                                        beta=-(min_v * 220.0 / (
                                                max_v - min_v) - 35))
            v_ret = v_ret[:, :, 0]
            v[mask] = v_ret[mask]
            img_ret[:, :, i] = v

    return img_ret


def color_inrange(img, color_space, hsv=None,
                  B_L=0, B_U=255,
                  G_L=0, G_U=255,
                  R_L=0, R_U=255,
                  H_L=0, H_U=179,
                  S_L=0, S_U=255,
                  V_L=0, V_U=255):
    if color_space == 'BGR':
        lower_range = np.array([B_L, G_L, R_L], dtype=np.uint8)
        upper_range = np.array([B_U, G_U, R_U], dtype=np.uint8)
        mask = cv2.inRange(img, lower_range, upper_range)
    elif color_space == 'HSV':
        if hsv is None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([H_L, S_L, V_L], dtype=np.uint8)
        upper_range = np.array([H_U, S_U, V_U], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_range, upper_range)
    return mask


def detect_color(img_hsv, color, on_surface=False):
    '''
    detect the area in @img_hsv with a specific @color, and return the @mask
    @img_hsv is the input in HSV color space
    @color is a string, describing color
    Currently supported colors: Black, White
    In OpenCV HSV space, H is in [0, 179], the other two are in [0, 255]
    '''
    if color == "black":
        mask1_1 = color_inrange(None, 'HSV', hsv=img_hsv[0], V_U=50)
        mask1_2 = color_inrange(None, 'HSV', hsv=img_hsv[1], S_U=60)
        mask1 = cv2.bitwise_and(mask1_1, mask1_2)
        mask2_1 = color_inrange(None, 'HSV', hsv=img_hsv[0], V_U=20)
        mask2_2 = color_inrange(None, 'HSV', hsv=img_hsv[1], S_U=100)
        mask2 = cv2.bitwise_and(mask2_1, mask2_2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "white":
        mask = color_inrange(None, 'HSV', hsv=img_hsv, S_U=60, V_L=190)
    else:
        print("ERROR: color detection has specified an undefined color!!!!")

    return mask


def detect_colors(img, mask_src, on_surface=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mask_src is None:
        mask_src = np.ones(img.shape[0:2], dtype=np.uint8) * 255
    mask_nothing = np.zeros(mask_src.shape, dtype=np.uint8)
    # detect green
    mask_green = color_inrange(img, 'HSV', hsv=hsv, H_L=45, H_U=96, S_L=80)
    mask_green = cv2.bitwise_and(mask_green, mask_src)
    mask_green_bool = mask_green.astype(bool)
    if np.any(mask_green_bool) and has_a_brick(mask_green, min_area=20,
                                               min_span=5):
        S_mean = np.median(hsv[mask_green_bool, 1])
        mask_green = color_inrange(img, 'HSV', hsv=hsv, H_L=45, H_U=96,
                                   S_L=int(S_mean * 0.7))
        if not has_a_brick(cv2.bitwise_and(mask_green, mask_src), min_area=20,
                           min_span=5):
            mask_green = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_green_bool, 2], 75)
            mask_green_on = color_inrange(img, 'HSV', hsv=hsv, H_L=45, H_U=96,
                                          S_L=int(S_mean * 0.7),
                                          V_L=V_ref * 0.75)
            mask_green = (mask_green, mask_green_on)
    else:
        mask_green = mask_nothing if not on_surface else (
            mask_nothing, mask_nothing)
    # detect yellow
    mask_yellow = color_inrange(img, 'HSV', hsv=hsv, H_L=8, H_U=45, S_L=90)
    mask_yellow = cv2.bitwise_and(mask_yellow, mask_src)
    mask_yellow_bool = mask_yellow.astype(bool)
    if np.any(mask_yellow_bool) and has_a_brick(mask_yellow, min_area=20,
                                                min_span=5):
        S_mean = np.median(hsv[mask_yellow_bool, 1])
        mask_yellow = color_inrange(img, 'HSV', hsv=hsv, H_L=8, H_U=45,
                                    S_L=int(S_mean * 0.7))
        if not has_a_brick(cv2.bitwise_and(mask_yellow, mask_src), min_area=20,
                           min_span=5):
            mask_yellow = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_yellow_bool, 2], 75)
            mask_yellow_on = color_inrange(img, 'HSV', hsv=hsv, H_L=8, H_U=45,
                                           S_L=int(S_mean * 0.7),
                                           V_L=V_ref * 0.75)
            mask_yellow = (mask_yellow, mask_yellow_on)
    else:
        mask_yellow = mask_nothing if not on_surface else (
            mask_nothing, mask_nothing)
    # detect red
    mask_red1 = color_inrange(img, 'HSV', hsv=hsv, H_L=0, H_U=10, S_L=105)
    mask_red2 = color_inrange(img, 'HSV', hsv=hsv, H_L=160, H_U=179, S_L=105)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.bitwise_and(mask_red, mask_src)
    mask_red_bool = mask_red.astype(bool)
    if np.any(mask_red_bool) and has_a_brick(mask_red, min_area=20, min_span=5):
        S_mean = np.median(hsv[mask_red_bool, 1])
        mask_red1 = color_inrange(img, 'HSV', hsv=hsv, H_L=0, H_U=10,
                                  S_L=int(S_mean * 0.7))
        mask_red2 = color_inrange(img, 'HSV', hsv=hsv, H_L=160, H_U=179,
                                  S_L=int(S_mean * 0.7))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        if not has_a_brick(cv2.bitwise_and(mask_red, mask_src), min_area=20,
                           min_span=5):
            mask_red = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_red_bool, 2], 75)
            mask_red1_on = color_inrange(img, 'HSV', hsv=hsv, H_L=0, H_U=10,
                                         S_L=int(S_mean * 0.7),
                                         V_L=V_ref * 0.75)
            mask_red2_on = color_inrange(img, 'HSV', hsv=hsv, H_L=160, H_U=179,
                                         S_L=int(S_mean * 0.7),
                                         V_L=V_ref * 0.75)
            mask_red_on = cv2.bitwise_or(mask_red1_on, mask_red2_on)
            mask_red = (mask_red, mask_red_on)
    else:
        mask_red = mask_nothing if not on_surface else (
            mask_nothing, mask_nothing)
    # detect blue
    mask_blue = color_inrange(img, 'HSV', hsv=hsv, H_L=93, H_U=140, S_L=125)
    mask_blue = cv2.bitwise_and(mask_blue, mask_src)
    mask_blue_bool = mask_blue.astype(bool)
    if np.any(mask_blue_bool) and has_a_brick(mask_blue, min_area=20,
                                              min_span=5):
        S_mean = np.median(hsv[mask_blue_bool, 1])
        mask_blue = color_inrange(img, 'HSV', hsv=hsv, H_L=93, H_U=140,
                                  S_L=int(S_mean * 0.8))
        if not has_a_brick(cv2.bitwise_and(mask_blue, mask_src), min_area=20,
                           min_span=5):
            mask_blue = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_blue_bool, 2], 75)
            mask_blue_on = color_inrange(img, 'HSV', hsv=hsv, H_L=93, H_U=140,
                                         S_L=int(S_mean * 0.8),
                                         V_L=V_ref * 0.75)
            mask_blue = (mask_blue, mask_blue_on)
    else:
        mask_blue = mask_nothing if not on_surface else (
            mask_nothing, mask_nothing)

    return (mask_green, mask_red, mask_yellow, mask_blue)


def detect_colorful(img, on_surface=False):
    lower_bound = [0, 100, 20]
    upper_bound = [179, 255, 255]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = np.array(lower_bound, dtype=np.uint8)
    upper_range = np.array(upper_bound, dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    return mask


##################### Some major functions #########################
def _locate_board(img, display_list):
    ## Find some black
    DoB = zc.get_DoB(img, config.BLUR_KERNEL_SIZE, 1, method='Average')
    zc.check_and_display('DoB', DoB, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    mask_black = color_inrange(DoB, 'HSV', V_L=config.BLACK_DOB_MIN_V)
    zc.check_and_display('mask_black', mask_black, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    ## 1. find black dots (somewhat black, and small)
    ## 2. find area where black dots density is high
    if 'mask_black_dots' in display_list:
        mask_black_dots = np.zeros(mask_black.shape, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask_black, mode=cv2.RETR_CCOMP,
                                           method=cv2.CHAIN_APPROX_NONE)
    bd_counts = np.zeros((config.BD_COUNT_N_ROW,
                          config.BD_COUNT_N_COL))  # count black dots in each
    # block
    for cnt_idx, cnt in enumerate(contours):
        if len(cnt) > config.BD_MAX_PERI or (hierarchy[0, cnt_idx, 3] != -1):
            continue
        if config.CHECK_BD_SIZE == 'complete':
            max_p = cnt.max(axis=0)
            min_p = cnt.min(axis=0)
            diff_p = max_p - min_p
            if diff_p.max() > config.BD_MAX_SPAN:
                continue
        mean_p = cnt.mean(axis=0)[0]
        bd_counts[int(mean_p[1] / config.BD_BLOCK_HEIGHT), int(
            mean_p[0] / config.BD_BLOCK_WIDTH)] += 1
        if 'mask_black_dots' in display_list:
            cv2.drawContours(mask_black_dots, contours, cnt_idx, 255, -1)
    if 'mask_black_dots' in display_list:
        zc.display_image('mask_black_dots', mask_black_dots,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    ## find a point that we are confident is in the board
    # print bd_counts
    max_idx = bd_counts.argmax()
    i, j = zc.ind2sub((config.BD_COUNT_N_ROW, config.BD_COUNT_N_COL), max_idx)
    if bd_counts[i, j] < config.BD_COUNT_THRESH:
        rtn_msg = {'status' : 'fail',
                   'message': 'Too little black dots, maybe image blurred'}
        # return (rtn_msg, None, None, None)
    in_board_p = (
        (i + 0.5) * config.BD_BLOCK_HEIGHT, (j + 0.5) * config.BD_BLOCK_WIDTH)

    ## locate the board by finding the contour that is likely to be of the board
    closest_cnt = zc.get_closest_contour(contours, hierarchy, in_board_p,
                                         min_span=config.BD_BLOCK_SPAN,
                                         hierarchy_req='inner')
    if closest_cnt is None or (not zc.is_roughly_convex(closest_cnt)):
        # rtn_msg = {'status' : 'fail',
        #           'message': 'Cannot locate board border, maybe not the full '
        #                      'board is in the scene. Failed at stage 1'}
        # return (rtn_msg, None, None, None)
        raise NoBoardDetectedError('Stage 1: Could not locate board border in '
                                   'frame.')

    hull = cv2.convexHull(closest_cnt)
    mask_board = np.zeros(mask_black.shape, dtype=np.uint8)
    cv2.drawContours(mask_board, [hull], 0, 255, -1)

    ## polish the board border in case the background is close to black
    cv2.drawContours(mask_board, [hull], 0, 255, 5)
    img_tmp = img.copy()
    img_tmp[np.invert(mask_board.astype(bool)), :] = 180
    DoB = zc.get_DoB(img_tmp, config.BLUR_KERNEL_SIZE, 1, method='Average')
    mask_black = color_inrange(DoB, 'HSV', V_L=config.BLACK_DOB_MIN_V)
    contours, hierarchy = cv2.findContours(mask_black, mode=cv2.RETR_CCOMP,
                                           method=cv2.CHAIN_APPROX_NONE)
    closest_cnt = zc.get_closest_contour(contours, hierarchy, in_board_p,
                                         min_span=config.BD_BLOCK_SPAN,
                                         hierarchy_req='inner')
    if closest_cnt is None or (not zc.is_roughly_convex(closest_cnt)):
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Cannot locate board border, maybe not the
        #            full '
        #                       'board is in the scene. Failed at stage 2'}
        # return (rtn_msg, None, None, None)
        raise NoBoardDetectedError('Stage 2: Could not locate board border in '
                                   'frame. ')

    hull = cv2.convexHull(closest_cnt)
    mask_board = np.zeros(mask_black.shape, dtype=np.uint8)
    cv2.drawContours(mask_board, [hull], 0, 255, -1)

    img_board = np.zeros(img.shape, dtype=np.uint8)
    img_board = cv2.bitwise_and(img, img, dst=img_board, mask=mask_board)
    zc.check_and_display('board_original', img_board, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    ## sanity checks
    if mask_board[int(in_board_p[0]), int(in_board_p[1])] == 0:
        # print(in_board_p)
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Best board candidate fails sanity check, '
        #                       'black dots are not inside the board...'}
        # # return (rtn_msg, None, None, None)
        raise NoBoardDetectedError('Best candidate failed sanity check.')

    return hull, mask_board, img_board


def _detect_lego(img_board, display_list, method='edge', edge_th=[80, 160],
                 mask_black_dots=None, mask_lego_rough=None, add_color=True):
    if method == 'edge':
        bw_board = cv2.cvtColor(img_board, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(bw_board, edge_th[0], edge_th[1], apertureSize=3)
        zc.check_and_display('board_edge', edges, display_list,
                             wait_time=config.DISPLAY_WAIT_TIME,
                             resize_max=config.DISPLAY_MAX_PIXEL,
                             save_image=config.SAVE_IMAGE)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 zc.generate_kernel(7, 'circular'),
                                 iterations=1)  # magic kernel size
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN,
                                 zc.generate_kernel(3, 'square'), iterations=1)
        mask_rough = cv2.bitwise_not(edges)
        if add_color:
            mask_color = detect_colorful(img_board)
            mask = cv2.bitwise_or(mask_rough, mask_color)
        else:
            mask = mask_rough
        zc.check_and_display('edge_inv', mask, display_list,
                             wait_time=config.DISPLAY_WAIT_TIME,
                             resize_max=config.DISPLAY_MAX_PIXEL,
                             save_image=config.SAVE_IMAGE)

    elif method == 'dots':
        mask = cv2.morphologyEx(mask_black_dots, cv2.MORPH_CLOSE,
                                zc.generate_kernel(11, 'square'),
                                iterations=1)  # magic kernel size
        mask = cv2.bitwise_not(mask)

    elif method == 'fill':  # This is not finished. Not working well with
        # initial tests. Don't use it.
        img = img_board.copy()
        mask_black_dots_bool = mask_black_dots.astype(bool)
        img[mask_black_dots_bool, :] = 0
        kernel = zc.generate_kernel(3, method='circular')
        for iter in range(1):
            img_tmp = cv2.dilate(img, kernel, iterations=1)
            img[mask_black_dots_bool] = img_tmp[mask_black_dots_bool]
            mask_black_dots = cv2.erode(mask_black_dots, kernel, iterations=1)
            mask_black_dots_bool = mask_black_dots.astype(bool)
        bw_board = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(bw_board, 100, 200, apertureSize=3)
        return img, edges

    # In case the large border of board is considered to be the best candidate
    if mask_lego_rough is not None:
        mask = cv2.bitwise_and(mask, mask_lego_rough)

    mask_lego, _ = zc.find_largest_CC(mask,
                                      min_area=config.BOARD_RECONSTRUCT_AREA
                                               / 300.0,
                                      min_convex_rate=0.2,
                                      ref_p=config.BOARD_RECONSTRUCT_CENTER,
                                      max_dist_ref_p=config.BOARD_RECONSTRUCT_PERI / 15.0)
    if mask_lego is None:
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Cannot find a large enough foreground near
        #            the '
        #                       'center of board'}
        # return (rtn_msg, None, None)
        raise NoLEGODetectedError('Cannot find a large enough foreground near '
                                  'the center of the board.')

    img_lego = np.zeros(img_board.shape, dtype=np.uint8)
    img_lego = cv2.bitwise_and(img_board, img_board, dst=img_lego,
                               mask=mask_lego)

    return img_lego, mask_lego


def _find_lego(img, stretch_ratio, display_list):
    ######################## detect board ######################################
    hull, mask_board, img_board = _locate_board(img, display_list)

    ## some properties of the board
    board_area = cv2.contourArea(hull)
    if board_area < config.BOARD_MIN_AREA:
        # rtn_msg = {'status': 'fail', 'message': 'Detected board too small'}
        # return (rtn_msg, None)
        raise NoBoardDetectedError('Detected board too small.')

    M = cv2.moments(hull)
    board_center = (
        int(M['m01'] / M['m00']),
        int(M['m10'] / M['m00']))  # in (row, col) format
    board_perimeter = cv2.arcLength(hull, True)
    # print "Board statistics: area: %d, center: %s, perimeter: %d" % (
    # board_area, board_center, board_perimeter)

    ## find the perspective correction matrix
    board_border = np.zeros(mask_board.shape, dtype=np.uint8)
    cv2.drawContours(board_border, [hull], 0, 255, 1)
    corners = get_corner_pts(board_border, board_perimeter, board_center,
                             method='line')
    if corners is None:
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Cannot locate exact four board corners, '
        #                       'probably because of occlusion'}
        # return (rtn_msg, None)
        raise NoBoardDetectedError('Cannot locate exact four board corners.')

    thickness = int(calc_thickness(corners,
                                   stretch_ratio) * 0.8)  # TODO: should be
    # able to be more accurate
    # print "Brick thickness: %d pixels" % thickness
    if config.OPT_FINE_BOARD:
        # first get a rough perspective matrix
        margin = config.BOARD_RECONSTRUCT_WIDTH // 5
        target_points = np.float32([[margin, margin],
                                    [config.BOARD_RECONSTRUCT_WIDTH + margin,
                                     margin], [margin,
                                               config.BOARD_RECONSTRUCT_HEIGHT + margin],
                                    [config.BOARD_RECONSTRUCT_WIDTH + margin,
                                     config.BOARD_RECONSTRUCT_HEIGHT + margin]])
        perspective_mtx = cv2.getPerspectiveTransform(corners, target_points)
        board_border = cv2.warpPerspective(board_border, perspective_mtx, (
            config.BOARD_RECONSTRUCT_WIDTH + margin * 2,
            config.BOARD_RECONSTRUCT_HEIGHT + margin * 2),
                                           flags=cv2.INTER_NEAREST)
        # fine adjustment to get more accurate perpective matrix
        corners = get_corner_pts(board_border, method='point')
        target_points = np.float32([[0, 0], [config.BOARD_RECONSTRUCT_WIDTH, 0],
                                    [0, config.BOARD_RECONSTRUCT_HEIGHT],
                                    [config.BOARD_RECONSTRUCT_WIDTH,
                                     config.BOARD_RECONSTRUCT_HEIGHT]])
        perspective_mtx2 = cv2.getPerspectiveTransform(corners, target_points)
        perspective_mtx = np.dot(perspective_mtx2, perspective_mtx)
    else:
        target_points = np.float32([[0, 0], [config.BOARD_RECONSTRUCT_WIDTH, 0],
                                    [0, config.BOARD_RECONSTRUCT_HEIGHT],
                                    [config.BOARD_RECONSTRUCT_WIDTH,
                                     config.BOARD_RECONSTRUCT_HEIGHT]])
        perspective_mtx = cv2.getPerspectiveTransform(corners, target_points)

    ## convert board to standard size for further processing
    img_board_original = img_board
    img_board = cv2.warpPerspective(img_board, perspective_mtx, (
        config.BOARD_RECONSTRUCT_WIDTH, config.BOARD_RECONSTRUCT_HEIGHT))
    zc.check_and_display('board', img_board, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    #################### detect Lego on the board ##############################
    ## locate Lego approach 1: using edges with pre-normalized image,
    # edge threshold is also pre-defined
    img_lego_u_edge_S, mask_lego_u_edge_S = _detect_lego(img_board,
                                                         display_list,
                                                         method='edge',
                                                         edge_th=[50,
                                                                  100],
                                                         add_color=False)
    mask_lego_rough_L = zc.expand(mask_lego_u_edge_S, 21, method='circular',
                                  iterations=2)
    mask_lego_rough_S = zc.expand(mask_lego_u_edge_S, 11, method='circular',
                                  iterations=2)
    mask_lego_rough_L_inv = cv2.bitwise_not(mask_lego_rough_L)
    mask_lego_rough_S_inv = cv2.bitwise_not(mask_lego_rough_S)

    ## correct color of board
    # find an area that should be grey in general
    # area where there are a lot of edges AND area far from the edges of board
    mask_grey = np.zeros(
        (config.BOARD_RECONSTRUCT_HEIGHT, config.BOARD_RECONSTRUCT_WIDTH),
        dtype=np.uint8)
    mask_grey[10: config.BOARD_RECONSTRUCT_HEIGHT - 10,
    50: config.BOARD_RECONSTRUCT_WIDTH - 60] = 255
    mask_board = np.zeros(
        (config.BOARD_RECONSTRUCT_HEIGHT, config.BOARD_RECONSTRUCT_WIDTH),
        dtype=np.uint8)
    mask_board[10: config.BOARD_RECONSTRUCT_HEIGHT - 10,
    10: config.BOARD_RECONSTRUCT_WIDTH - 10] = 255
    mask_grey = cv2.bitwise_and(mask_grey, mask_lego_rough_S_inv)
    mask_grey_bool = mask_grey.astype(bool)
    if not np.any(mask_grey_bool):
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Cannot find grey area, maybe image blurred'}
        # return (rtn_msg, None)
        raise NoLEGODetectedError('Cannot find grey area, blurry image?')
    if 'board_grey' in display_list:
        img_board_grey = np.zeros(img_board.shape, dtype=np.uint8)
        img_board_grey = cv2.bitwise_and(img_board, img_board,
                                         dst=img_board_grey, mask=mask_grey)
        zc.check_and_display('board_grey', img_board_grey, display_list,
                             wait_time=config.DISPLAY_WAIT_TIME,
                             resize_max=config.DISPLAY_MAX_PIXEL,
                             save_image=config.SAVE_IMAGE)

    ## locate Lego approach 1 continued: refinement by using auto selected
    # thresholds
    bw_board = cv2.cvtColor(img_board, cv2.COLOR_BGR2GRAY)
    dynamic_range = bw_board[mask_grey_bool].max() - bw_board[
        mask_grey_bool].min()
    edge_th = [dynamic_range / 4 + 35, dynamic_range / 2 + 70]
    img_lego_u_edge_S, mask_lego_u_edge_S = _detect_lego(img_board,
                                                         display_list,
                                                         method='edge',
                                                         edge_th=edge_th,
                                                         add_color=False)
    zc.check_and_display('lego_u_edge_S', img_lego_u_edge_S, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    ## locate Lego approach 2: using edges with normalized image
    # the three steps below for color correction is not well studied,
    # but it's here for historical reasons...
    img_board_n0 = normalize_color(img_board, mask_apply=mask_board,
                                   mask_info=mask_grey, method='grey')
    img_board_n0 = normalize_brightness(img_board_n0, mask=mask_board,
                                        method='max')
    img_board_n0 = normalize_color(img_board_n0, mask_apply=mask_board,
                                   mask_info=mask_grey, method='hist')
    zc.check_and_display('board_n0', img_board_n0, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    bw_board_n0 = cv2.cvtColor(img_board_n0, cv2.COLOR_BGR2GRAY)
    dynamic_range = bw_board_n0[mask_grey_bool].max() - bw_board_n0[
        mask_grey_bool].min()
    edge_th = [dynamic_range / 4 + 35, dynamic_range / 2 + 70]
    img_lego_u_edge_norm_L, mask_lego_u_edge_norm_L = _detect_lego(
        img_board_n0, display_list, method='edge', edge_th=edge_th,
        add_color=True)
    zc.check_and_display('lego_u_edge_norm_L', img_lego_u_edge_norm_L,
                         display_list, wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    ## locate Lego approach 3: using dots with pre-normalized image
    # black dot detection
    mask_lego_bool = mask_lego_u_edge_S.astype(bool)
    img_board_tmp = img_board.copy()
    img_board_tmp[mask_lego_bool, :] = (int(
        bw_board[mask_grey_bool].max()) + int(
        bw_board[mask_grey_bool].min())) / 2
    DoB = zc.get_DoB(img_board_tmp, 41, 1, method='Average')
    DoB[mask_lego_bool] = 0
    # DoB[mask_lego_rough_L_inv] = 0
    zc.check_and_display('board_DoB', DoB, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    mask_black = color_inrange(DoB, 'HSV', V_L=config.BD_DOB_MIN_V)
    zc.check_and_display('board_mask_black', mask_black, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    mask_black_dots, n_cnts = zc.get_small_blobs(mask_black,
                                                 max_peri=config.BOARD_BD_MAX_PERI)
    if n_cnts < 1000:  # some sanity check
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Too little black dots with more accurate dot '
        #                       'detection. Image may be blurred'}
        # return (rtn_msg, None)
        raise NoBoardDetectedError('Too few black dots, blurry image?')
    zc.check_and_display('board_mask_black_dots', mask_black_dots, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    img_lego_u_dots_L, mask_lego_u_dots_L = _detect_lego(img_board,
                                                         display_list,
                                                         method='dots',
                                                         mask_black_dots=mask_black_dots,
                                                         mask_lego_rough=mask_lego_rough_L,
                                                         add_color=False)

    zc.check_and_display('lego_u_dots_L', img_lego_u_dots_L, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    ## detect colors of Lego
    mask_no_black_dots = cv2.bitwise_and(mask_grey,
                                         cv2.bitwise_not(mask_black_dots))
    # correct color in different ways
    img_board_n1 = normalize_color(img_board, mask_apply=mask_board,
                                   mask_info=mask_grey, method='grey')
    img_board_n2 = normalize_brightness(img_board_n1, mask=mask_board,
                                        method='max', max_percentile=95,
                                        min_percentile=1)
    img_board_n3 = normalize_color(img_board, mask_apply=mask_board,
                                   mask_info=mask_black_dots, method='grey')
    img_board_n4 = normalize_brightness(img_board_n3, mask=mask_board,
                                        method='max', max_percentile=95,
                                        min_percentile=1)
    img_board_n5 = normalize_color(img_board, mask_apply=mask_board,
                                   mask_info=mask_no_black_dots, method='grey')
    img_board_n6 = normalize_brightness(img_board_n5, mask=mask_board,
                                        method='max', max_percentile=95,
                                        min_percentile=1)
    zc.check_and_display('board_n1', img_board_n1, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    zc.check_and_display('board_n2', img_board_n2, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    zc.check_and_display('board_n3', img_board_n3, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    zc.check_and_display('board_n4', img_board_n4, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    zc.check_and_display('board_n5', img_board_n5, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    zc.check_and_display('board_n6', img_board_n6, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    mask_green, mask_red, mask_yellow, mask_blue = detect_colors(img_board,
                                                                 mask_lego_u_edge_S)
    mask_green_n1, mask_red_n1, mask_yellow_n1, mask_blue_n1 = detect_colors(
        img_board_n1, mask_lego_u_edge_S)
    mask_green_n3, mask_red_n3, mask_yellow_n3, mask_blue_n3 = detect_colors(
        img_board_n3, mask_lego_u_edge_S)
    mask_green = zc.super_bitwise_and(
        (mask_green, mask_green_n1, mask_green_n3, mask_lego_u_edge_norm_L))
    mask_yellow = zc.super_bitwise_and(
        (mask_yellow, mask_yellow_n1, mask_yellow_n3, mask_lego_u_edge_norm_L))
    mask_red = zc.super_bitwise_and(
        (mask_red, mask_red_n1, mask_red_n3, mask_lego_u_edge_norm_L))
    mask_blue = zc.super_bitwise_and(
        (mask_blue, mask_blue_n1, mask_blue_n3, mask_lego_u_edge_norm_L))
    if 'lego_only_color' in display_list:
        color_labels = np.zeros(img_board.shape[0:2], dtype=np.uint8)
        color_labels[mask_green.astype(bool)] = 2
        color_labels[mask_yellow.astype(bool)] = 3
        color_labels[mask_red.astype(bool)] = 4
        color_labels[mask_blue.astype(bool)] = 5
        img_color = bm.bitmap2syn_img(color_labels)
        zc.check_and_display('lego_only_color', img_color, display_list,
                             wait_time=config.DISPLAY_WAIT_TIME,
                             resize_max=config.DISPLAY_MAX_PIXEL,
                             save_image=config.SAVE_IMAGE)
    mask_green = zc.expand_with_bound(mask_green, mask_lego_u_dots_L)
    mask_yellow = zc.expand_with_bound(mask_yellow, mask_lego_u_dots_L)
    mask_red = zc.expand_with_bound(mask_red, mask_lego_u_dots_L)
    mask_blue = zc.expand_with_bound(mask_blue, mask_lego_u_dots_L)

    ## add colorful parts to the Lego to get Lego with all sides
    mask_lego_full = zc.super_bitwise_or(
        (mask_green, mask_yellow, mask_red, mask_blue, mask_lego_u_edge_S))
    mask_lego_full, _ = zc.find_largest_CC(mask_lego_full,
                                           min_area=config.BOARD_RECONSTRUCT_AREA / 300.0,
                                           min_convex_rate=0.2,
                                           ref_p=config.BOARD_RECONSTRUCT_CENTER,
                                           max_dist_ref_p=config.BOARD_RECONSTRUCT_PERI / 15.0)
    if mask_lego_full is None:
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Cannot find a large enough foreground near
        #            the '
        #                       'center of the board after adding all colors '
        #                       'back to Lego'}
        # return (rtn_msg, None, None)
        raise NoLEGODetectedError(
            'Cannot find a large enough foreground near the '
            'center of the board after adding all colors back to Lego')

    img_lego_full = np.zeros(img_board.shape, dtype=np.uint8)
    img_lego_full = cv2.bitwise_and(img_board, img_board, dst=img_lego_full,
                                    mask=mask_lego_full)
    zc.check_and_display('lego_full', img_lego_full, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    ## erode side parts in original view
    img_lego_full_original = cv2.warpPerspective(img_lego_full, perspective_mtx,
                                                 img.shape[1::-1],
                                                 flags=cv2.WARP_INVERSE_MAP)
    mask_lego_full_original = zc.get_mask(img_lego_full_original)
    # treat white brick differently to prevent it from erosion
    hsv_lego = cv2.cvtColor(img_lego_full_original, cv2.COLOR_BGR2HSV)
    mask_lego_white = detect_color(hsv_lego, 'white')
    mask_lego_white, _ = zc.get_big_blobs(mask_lego_white, min_area=25)
    kernel = np.uint8([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    mask_lego = cv2.erode(mask_lego_full_original, kernel, iterations=thickness)
    mask_lego = cv2.bitwise_or(mask_lego, mask_lego_white)
    mask_lego, _ = zc.find_largest_CC(mask_lego)
    if mask_lego is None:
        # rtn_msg = {'status' : 'fail',
        #            'message': 'Cannot find Lego on the board after eroding '
        #                       'side parts'}
        # return (rtn_msg, None)
        raise NoLEGODetectedError('Cannot find Lego on the board after '
                                  'eroding side parts.')
    img_lego = np.zeros(img.shape, dtype=np.uint8)
    img_lego = cv2.bitwise_and(img, img, dst=img_lego,
                               mask=mask_lego)  # this is weird, if not
    # providing an input image, the output will be with random backgrounds...
    # how is dst initialized?
    img_lego = cv2.warpPerspective(img_lego, perspective_mtx, (
        config.BOARD_RECONSTRUCT_WIDTH, config.BOARD_RECONSTRUCT_HEIGHT))
    zc.check_and_display('lego', img_lego, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    return (img_lego, img_lego_full, img_board, (
        img_board_n0, img_board_n1, img_board_n2, img_board_n3, img_board_n4,
        img_board_n5, img_board_n6), perspective_mtx)


def _find_lego_noboard(img, stretch_ratio, display_list):
    '''
    Find lego without the help of board (e.g. put the lego pieces directly on
    the table.
    Not finished.
    '''
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(bw, 50, 100, apertureSize=3)
    zc.check_and_display('edge', edges, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)
    edges_dilated = zc.expand(edges, 3)

    mask_lego = np.zeros(img.shape, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(edges_dilated, mode=cv2.RETR_CCOMP,
                                           method=cv2.CHAIN_APPROX_NONE)
    for cnt_idx, cnt in enumerate(contours):
        if hierarchy[0, cnt_idx, 3] == -1:
            continue
        mask_tmp = np.zeros(bw.shape, dtype=np.uint8)
        cv2.drawContours(mask_tmp, contours, cnt_idx, 255, -1)
        mask_tmp = zc.expand(mask_tmp, 3)
        contours_tmp, hierarchy_tmp = cv2.findContours(mask_tmp,
                                                       mode=cv2.RETR_CCOMP,
                                                       method=cv2.CHAIN_APPROX_NONE)
        cnt_tmp = contours_tmp[0]
        mask_tmp = np.zeros(bw.shape, dtype=np.uint8)
        cv2.drawContours(mask_tmp, [cnt_tmp], 0, 255, 1)
        cv2.drawContours(mask_lego, contours, cnt_idx, [255, 255, 255], -1)

        lines = cv2.HoughLinesP(mask_tmp, 1, np.pi / 180, 3, minLineLength=4,
                                maxLineGap=1)
        if lines is None:
            continue
        lines = lines[0]
        for line in lines:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(mask_lego, pt1, pt2, (0, 0, 255), 1)

        line_groups = []
        for line in lines:
            merge_flag = False
            for line_group in line_groups:
                print((is_line_seg_close(line_group, line),
                       is_line_seg_close2(line_group, line)))
                if is_line_seg_close(line_group, line) and is_line_seg_close(
                        line_group, line):
                    merge_flag = True
                    line_group_new = line_group.copy()
                    line_group_new[0] = min(line_group[0], line_group[2],
                                            line[0], line[2])
                    line_group_new[1] = min(line_group[1], line_group[3],
                                            line[1], line[3])
                    line_group_new[2] = max(line_group[0], line_group[2],
                                            line[0], line[2])
                    line_group_new[3] = max(line_group[1], line_group[3],
                                            line[1], line[3])
                    line_group[0] = line_group_new[0]
                    line_group[1] = line_group_new[1]
                    line_group[2] = line_group_new[2]
                    line_group[3] = line_group_new[3]
            if not merge_flag:
                line_groups.append(line)
        for line in line_groups:
            pt1 = (line[0], line[1])
            pt2 = (line[2], line[3])
            cv2.line(mask_lego, pt1, pt2, (0, 255, 0), 1)

    zc.check_and_display('edge_inv', mask_lego, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    rtn_msg = {'status': 'fail', 'message': 'nothing'}
    return (rtn_msg, None)


def _correct_orientation(img_lego, img_lego_full, display_list):
    objects = rotate(img_lego)
    img_lego_correct, rotation_degree, rotation_mtx = objects
    objects = rotate(img_lego_full)
    img_lego_full_correct, rotation_degree_full, rotation_mtx = objects

    # print (rotation_degree, rotation_degree_full)
    rotation_degree = rotation_degree * 0.6 + rotation_degree_full * 0.4
    rotation_mtx = cv2.getRotationMatrix2D(
        (img_lego.shape[1] / 2, img_lego.shape[0] / 2), rotation_degree,
        scale=1)
    img_lego_correct = cv2.warpAffine(img_lego, rotation_mtx,
                                      (img_lego.shape[1], img_lego.shape[0]))
    img_lego_full_correct = cv2.warpAffine(img_lego_full, rotation_mtx, (
        img_lego.shape[1], img_lego.shape[0]))
    zc.check_and_display('lego_correct', img_lego_correct, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    return img_lego_correct, img_lego_full_correct, rotation_mtx


def _get_rectangular_area(img_board, img_correct, rotation_mtx, display_list):
    img_shape = img_correct.shape
    img_cropped, borders = crop(img_correct, None)
    min_row, max_row, min_col, max_col = borders
    mask_rect = np.zeros(img_correct.shape[0:2], dtype=np.uint8)
    mask_rect[min_row: max_row + 1, min_col: max_col + 1] = 255
    mask_rect = cv2.warpAffine(mask_rect, rotation_mtx,
                               (img_shape[1], img_shape[0]),
                               flags=cv2.WARP_INVERSE_MAP)

    img_lego_rect = np.zeros(img_board.shape, dtype=np.uint8)
    img_lego_rect = cv2.bitwise_and(img_board, img_board, dst=img_lego_rect,
                                    mask=mask_rect)
    img_lego_rect = cv2.warpAffine(img_lego_rect, rotation_mtx,
                                   (img_shape[1], img_shape[0]))

    zc.check_and_display('lego_rect', img_lego_rect, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_max=config.DISPLAY_MAX_PIXEL,
                         save_image=config.SAVE_IMAGE)

    rtn_msg = {'status': 'success'}
    return (rtn_msg, img_lego_rect)


def _img2bitmap(img, color_cumsums, n_rows, n_cols, lego_color):
    height, width, _ = img.shape
    img_plot = None
    bitmap = np.zeros((n_rows, n_cols), dtype=int)
    best_ratio = 0
    best_bitmap = None
    best_plot = None
    best_offset = None

    offset_range = {'t': 0,
                    'b': int(round(config.BRICK_HEIGHT / 3)),
                    'l': int(round(config.BRICK_WIDTH / 3)),
                    'r': int(round(config.BRICK_WIDTH / 3))}

    for height_offset_t in range(0, offset_range['t'] + 1, 2):
        for height_offset_b in range(0, offset_range['b'] + 1, 2):
            for width_offset_l in range(0, offset_range['l'] + 1, 2):
                for width_offset_r in range(0, offset_range['r'] + 1, 2):
                    if 'plot_line' in config.DISPLAY_LIST:
                        if lego_color is not None:
                            img_plot = lego_color.copy()
                        else:
                            img_plot = img.copy()

                    test_height = height - height_offset_t - height_offset_b
                    test_width = width - width_offset_l - width_offset_r
                    block_height = float(test_height) / n_rows
                    block_width = float(test_width) / n_cols
                    n_pixels = test_height * test_width
                    n_pixels_center = 0
                    n_good_pixels = 0
                    n_good_pixels_center = 0
                    worst_ratio_block = 1  # set to maximum
                    for i in range(n_rows):
                        i_start = int(round(block_height * i)) + height_offset_t
                        i_end = int(
                            round(block_height * (i + 1))) + height_offset_t
                        for j in range(n_cols):
                            j_start = int(
                                round(block_width * j)) + width_offset_l
                            j_end = int(
                                round(block_width * (j + 1))) + width_offset_l
                            if 'plot_line' in config.DISPLAY_LIST:
                                cv2.line(img_plot, (j_end, 0),
                                         (j_end, height - 1), (255, 255, 0), 1)
                                cv2.line(img_plot, (0, i_end),
                                         (width - 1, i_end), (255, 255, 0), 1)
                                cv2.line(img_plot, (j_start, 0),
                                         (j_start, height - 1), (255, 255, 0),
                                         1)
                                cv2.line(img_plot, (0, i_start),
                                         (width - 1, i_start), (255, 255, 0), 1)
                            color_sum = {}
                            for color_key, color_cumsum in \
                                    list(color_cumsums.items()):
                                # focus more on center part
                                color_sum[color_key] = \
                                    color_cumsum[
                                        i_end - config.BLOCK_DETECTION_OFFSET,
                                        j_end - config.BLOCK_DETECTION_OFFSET] \
                                    - color_cumsum[
                                        i_start + config.BLOCK_DETECTION_OFFSET,
                                        j_end - config.BLOCK_DETECTION_OFFSET] \
                                    - color_cumsum[
                                        i_end - config.BLOCK_DETECTION_OFFSET,
                                        j_start +
                                        config.BLOCK_DETECTION_OFFSET] \
                                    + color_cumsum[
                                        i_start + config.BLOCK_DETECTION_OFFSET,
                                        j_start + config.BLOCK_DETECTION_OFFSET]

                            counts = [color_sum['nothing'], color_sum['white'],
                                      color_sum['green'], color_sum['yellow'],
                                      color_sum['red'], color_sum['blue'],
                                      color_sum['black'], color_sum['unsure']]
                            color_idx = np.argmax(counts[:-1])
                            bitmap[i, j] = color_idx
                            # percentage correct for center part of block
                            n_pixels_block_center = sum(counts)
                            ratio_block_center = \
                                float(counts[color_idx]) / n_pixels_block_center
                            n_pixels_center += n_pixels_block_center
                            n_good_pixels_center += counts[color_idx]

                            color_cumsum = color_cumsums[
                                config.COLOR_ORDER[color_idx]]
                            n_good_pixels_block = color_cumsum[i_end, j_end] \
                                                  - color_cumsum[i_start,
                                                                 j_end] \
                                                  - color_cumsum[i_end,
                                                                 j_start] \
                                                  + color_cumsum[
                                                      i_start, j_start]
                            color_cumsum = color_cumsums['unsure']
                            n_good_pixels_block += \
                                (color_cumsum[i_end, j_end]
                                 - color_cumsum[i_start, j_end]
                                 - color_cumsum[i_end, j_start]
                                 + color_cumsum[i_start, j_start]) / 2.0
                            # unsure pixels are half right
                            n_good_pixels += n_good_pixels_block
                            n_pixels_block = \
                                (j_end - j_start) * (i_end - i_start)
                            # percentage correct for entire block
                            ratio_block = n_good_pixels_block / n_pixels_block
                            if config.OPT_NOTHING and color_idx == 0:
                                ratio_block *= 0.9

                            ratio_block = ratio_block * 0.34 + \
                                          ratio_block_center * 0.66
                            if ratio_block < worst_ratio_block:
                                worst_ratio_block = ratio_block
                    ratio = float(n_good_pixels) / n_pixels * 0.34 + float(
                        n_good_pixels_center) / n_pixels_center * 0.66
                    # print "worst ratio within block: %f" % worst_ratio_block
                    if worst_ratio_block > config.WORST_RATIO_BLOCK_THRESH \
                            and ratio > best_ratio:
                        best_ratio = ratio
                        best_bitmap = bitmap.copy()
                        best_plot = img_plot
                        best_offset = (
                            height_offset_t, height_offset_b, width_offset_l,
                            width_offset_r)

    return best_bitmap, best_ratio, best_plot, best_offset


def _reconstruct_lego(img_lego, img_board, img_board_ns, rotation_mtx,
                      display_list):
    def _lego_outof_board(mask_lego, img_board, rotation_mtx, borders):
        img_lego = np.zeros(img_board.shape, dtype=np.uint8)
        img_lego = cv2.bitwise_and(img_board, img_board, dst=img_lego,
                                   mask=mask_lego)
        img_lego = cv2.warpAffine(img_lego, rotation_mtx,
                                  (img_board.shape[1], img_board.shape[0]))
        if borders is None:
            img_lego, borders = crop(img_lego, None)
            min_row, max_row, min_col, max_col = borders
            img_lego, borders = smart_crop(img_lego)
            i_start, i_end, j_start, j_end = borders
            borders = (min_row + i_start, min_row + i_end, min_col + j_start,
                       min_col + j_end)
        else:
            img_lego, borders = crop(img_lego, borders)
        return img_lego, borders

    ## get Lego images that are color-corrected in different ways
    img_board_n0, img_board_n1, img_board_n2, img_board_n3, img_board_n4, \
    img_board_n5, img_board_n6 = img_board_ns
    mask_lego = zc.get_mask(img_lego)
    img_lego, borders = _lego_outof_board(mask_lego, img_board, rotation_mtx,
                                          None)
    img_lego_n0, borders = _lego_outof_board(mask_lego, img_board_n0,
                                             rotation_mtx, borders)
    img_lego_n1, borders = _lego_outof_board(mask_lego, img_board_n1,
                                             rotation_mtx, borders)
    img_lego_n2, borders = _lego_outof_board(mask_lego, img_board_n2,
                                             rotation_mtx, borders)
    img_lego_n3, borders = _lego_outof_board(mask_lego, img_board_n3,
                                             rotation_mtx, borders)
    img_lego_n4, borders = _lego_outof_board(mask_lego, img_board_n4,
                                             rotation_mtx, borders)
    img_lego_n5, borders = _lego_outof_board(mask_lego, img_board_n5,
                                             rotation_mtx, borders)
    img_lego_n6, borders = _lego_outof_board(mask_lego, img_board_n6,
                                             rotation_mtx, borders)
    zc.check_and_display('lego_cropped', img_lego, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_scale=config.DISPLAY_SCALE,
                         save_image=config.SAVE_IMAGE)

    ## detect colors: green, red, yellow, blue
    mask_green, mask_red, mask_yellow, mask_blue = detect_colors(img_lego, None,
                                                                 on_surface=True)
    mask_green_n1, mask_red_n1, mask_yellow_n1, mask_blue_n1 = detect_colors(
        img_lego_n1, None, on_surface=True)
    mask_green_n3, mask_red_n3, mask_yellow_n3, mask_blue_n3 = detect_colors(
        img_lego_n3, None, on_surface=True)
    mask_green_on = zc.super_bitwise_and(
        (mask_green[1], mask_green_n1[1], mask_green_n3[1]))
    mask_yellow_on = zc.super_bitwise_and(
        (mask_yellow[1], mask_yellow_n1[1], mask_yellow_n3[1]))
    mask_red_on = zc.super_bitwise_and(
        (mask_red[1], mask_red_n1[1], mask_red_n3[1]))
    mask_blue_on = zc.super_bitwise_and(
        (mask_blue[1], mask_blue_n1[1], mask_blue_n3[1]))
    mask_green_all = zc.super_bitwise_and(
        (mask_green[0], mask_green_n1[0], mask_green_n3[0]))
    mask_yellow_all = zc.super_bitwise_and(
        (mask_yellow[0], mask_yellow_n1[0], mask_yellow_n3[0]))
    mask_red_all = zc.super_bitwise_and(
        (mask_red[0], mask_red_n1[0], mask_red_n3[0]))
    mask_blue_all = zc.super_bitwise_and(
        (mask_blue[0], mask_blue_n1[0], mask_blue_n3[0]))
    mask_colors = zc.super_bitwise_or(
        (mask_green_all, mask_yellow_all, mask_red_all, mask_blue_all))
    mask_colors_inv = cv2.bitwise_not(mask_colors)

    ## detect black and white
    hsv_lego_dark = cv2.cvtColor(img_lego_n4, cv2.COLOR_BGR2HSV)
    hsv_lego_bright = cv2.cvtColor(img_lego_n3, cv2.COLOR_BGR2HSV)
    mask_black = detect_color((hsv_lego_dark, hsv_lego_bright), 'black')
    hsv_lego = cv2.cvtColor(img_lego_n6, cv2.COLOR_BGR2HSV)
    mask_white = detect_color(hsv_lego, 'white')
    mask_black = cv2.bitwise_and(mask_black, mask_colors_inv)
    mask_white = cv2.bitwise_and(mask_white, mask_colors_inv)
    white, green, red, yellow, blue, black = zc.mask2bool((mask_white,
                                                           mask_green_on,
                                                           mask_red_on,
                                                           mask_yellow_on,
                                                           mask_blue_on,
                                                           mask_black))
    nothing = np.bitwise_and(
        np.bitwise_and(img_lego[:, :, 0] == 0, img_lego[:, :, 1] == 0),
        img_lego[:, :, 2] == 0)
    black = np.bitwise_and(black, np.invert(nothing))
    unsure = np.invert(
        zc.super_bitwise_or((nothing, white, green, red, yellow, blue, black)))

    ## calculate cumulative sum for color pixels to speed up sum operation
    nothing_cumsum = zc.calc_cumsum(nothing)
    white_cumsum = zc.calc_cumsum(white)
    green_cumsum = zc.calc_cumsum(green)
    yellow_cumsum = zc.calc_cumsum(yellow)
    red_cumsum = zc.calc_cumsum(red)
    blue_cumsum = zc.calc_cumsum(blue)
    black_cumsum = zc.calc_cumsum(black)
    unsure_cumsum = zc.calc_cumsum(unsure)
    color_cumsums = {'nothing': nothing_cumsum,
                     'white'  : white_cumsum,
                     'green'  : green_cumsum,
                     'yellow' : yellow_cumsum,
                     'red'    : red_cumsum,
                     'blue'   : blue_cumsum,
                     'black'  : black_cumsum,
                     'unsure' : unsure_cumsum
                     }
    # generate an image with each pixel as its assigned color, for debug purpose
    lego_color = None
    if 'lego_color' in display_list:
        color_labels = np.zeros(nothing.shape, dtype=np.uint8)
        color_labels[white] = 1
        color_labels[green] = 2
        color_labels[yellow] = 3
        color_labels[red] = 4
        color_labels[blue] = 5
        color_labels[black] = 6
        color_labels[unsure] = 7
        lego_color = bm.bitmap2syn_img(color_labels)
        zc.check_and_display('lego_color', lego_color, display_list,
                             wait_time=config.DISPLAY_WAIT_TIME,
                             resize_scale=config.DISPLAY_SCALE,
                             save_image=config.SAVE_IMAGE)

    ## real stuff begins...
    height, width, _ = img_lego.shape
    # print "Expected rows and cols: %f, %f" % (height / config.BRICK_HEIGHT,
    # width / config.BRICK_WIDTH)
    # calculate candidate number of rows and columns for the Lego model,
    # which is purely based on the pixel numbers
    n_rows_opt = max(int((height / config.BRICK_HEIGHT) + 0.5), 1)
    n_cols_opt = max(int((width / config.BRICK_WIDTH) + 0.3), 1)
    best_ratio = 0
    best_bitmap = None
    best_plot = None
    # best_offset = None

    # the real number of rows and columns may be slightly different, but we
    # don't consider it now...
    for n_rows in range(n_rows_opt - 0, n_rows_opt + 1):
        for n_cols in range(n_cols_opt - 0, n_cols_opt + 1):
            bitmap, ratio, img_plot, _ = _img2bitmap(img_lego, color_cumsums,
                                                     n_rows, n_cols, lego_color)
            if bitmap is None:
                continue
            # print "Reconstruction confidence: %f" % ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_bitmap = bitmap
                best_plot = img_plot
    if best_bitmap is None \
            or best_ratio < config.BRICK_MIN_BM_RATIO \
            or best_bitmap.shape != (n_rows_opt, n_cols_opt):
        raise LEGOCVError('Not confident about reconstruction, '
                          f'maybe too much noise. Best ratio: {best_ratio}')
    zc.check_and_display('plot_line', best_plot, display_list,
                         wait_time=config.DISPLAY_WAIT_TIME,
                         resize_scale=config.DISPLAY_SCALE,
                         save_image=config.SAVE_IMAGE)

    return best_bitmap


def process(img, stretch_ratio, display_list):
    objects = _find_lego(img, stretch_ratio, display_list)
    img_lego, img_lego_full, img_board, img_board_ns, perspective_mtx = \
        objects
    objects = _correct_orientation(img_lego, img_lego_full,
                                   display_list)

    img_lego_correct, img_lego_full_correct, rotation_mtx = objects

    return _reconstruct_lego(img_lego, img_board, img_board_ns,
                             rotation_mtx, display_list)
