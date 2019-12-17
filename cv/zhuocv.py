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

'''
This is a simple library file for common CV tasks
'''

import cv2
import math
import numpy as np
import os
import sys
import time

current_milli_time = lambda: int(round(time.time() * 1000))

################################ BASICS ########################################
def ind2sub(size, idx):
    '''
    Convert an index to a tuple of (row_idx, col_idx)
    @size is the size of the image: (n_rows, n_cols)
    '''
    return (idx / size[1], idx % size[1])

def euc_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def color_dist(img, ref_color):
    img_tmp = img.astype(int)
    img_tmp[:, :, 0] -= ref_color[0]
    img_tmp[:, :, 1] -= ref_color[1]
    img_tmp[:, :, 2] -= ref_color[2]
    dist = np.sqrt(np.sum(img_tmp ** 2, axis = 2))
    return dist

def angle_dist(a1, a2, angle_range = 180):
    dist1 = a2 - a1
    if dist1 > 0:
        dist2 = a2 - angle_range - a1
    else:
        dist2 = a2 + angle_range - a1
    return dist1 if abs(dist2) > abs(dist1) else dist2


def line_angle(p1, p2, reference = "x"):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]

    return math.atan2(delta_y, delta_x)

def super_bitwise_or(masks):
    final_mask = None
    for mask in masks:
        if final_mask is None:
            final_mask = mask
            continue
        final_mask = np.bitwise_or(final_mask, mask)
    return final_mask

def super_bitwise_and(masks):
    final_mask = None
    for mask in masks:
        if final_mask is None:
            final_mask = mask
            continue
        final_mask = np.bitwise_and(final_mask, mask)
    return final_mask

def generate_kernel(size, method = 'square'):
    kernel = None
    if method == 'square':
        kernel = np.ones((size, size), np.uint8)
    elif method == 'circular':
        y, x = np.ogrid[0:size, 0:size]
        center = (size / 2.0 - 0.5, size / 2.0 - 0.5)
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= (size / 2.0) ** 2
        kernel = mask.astype(np.uint8) * 255
    return kernel

def expand(img, size, method = 'square', iterations = 1):
    kernel = generate_kernel(size, method = method)
    return cv2.dilate(img, kernel, iterations = iterations)

def shrink(img, size, method = 'square', iterations = 1):
    kernel = generate_kernel(size, method = method)
    return cv2.erode(img, kernel, iterations = iterations)

def raw2cv_image(raw_data, gray_scale = False):
    img_array = np.asarray(bytearray(raw_data), dtype=np.int8)
    if gray_scale:
        cv_image = cv2.imdecode(img_array, 0)
    else:
        cv_image = cv2.imdecode(img_array, -1)
    return cv_image

def cv_image2raw(img, jpeg_quality = 95):
    result, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    raw_data = data.tostring()
    return raw_data

def mask2bool(masks):
    bools = []
    for mask in masks:
        #mask[mask == 255] = 1
        mask = mask.astype(bool)
        bools.append(mask)
    return bools

def get_mask(img, rtn_type = "mask", th = 0):
    '''
    Given a color or black-white image, return the mask where the pixels are non-zero.
    '''
    img_shape = img.shape
    if len(img_shape) > 2 and img_shape[2] > 1: # color image
        mask = np.zeros(img_shape[0:2], dtype = bool)
        for i in xrange(img_shape[2]):
            mask = np.bitwise_or(mask, img[:,:,i] > th)
    else:
        mask = img > th
    if rtn_type == "bool":
        return mask
    else:
        return mask.astype(np.uint8) * 255

def get_edge_point(mask, direction):
    '''
    Given a @mask, find the extreme point along given direction.
    Returns the found point.
    If there are multiple points at the end of the direction, returns one of them.
    Returns None if @mask is empty.
    @direction is a vector. For example, if direction == (1, -1), then returns the most upper right point.
    Notice that in a picture, y axis points toward downside.
    '''
    nonzero = np.nonzero(mask)
    if len(nonzero) < 2 or len(nonzero[0]) == 0: # mask is empty
        return None
    rows, cols = nonzero
    mix = direction[0] * cols + direction[1] * rows
    idx = np.argmax(mix)
    p = (cols[idx], rows[idx])

    return p

def get_DoB(img, k1, k2, method = 'Average'):
    '''
    Get difference of blur of an image (@img) with @method.
    The two blurred image are with kernal size @k1 and @k2.
    @method can be one of the strings: 'Gaussian', 'Average'.
    '''
    if k1 == 1:
        blurred1 = img
    elif method == 'Gaussian':
        blurred1 = cv2.GaussianBlur(img, (k1, k1), 0)
    elif method == 'Average':
        blurred1 = cv2.blur(img, (k1, k1))
    if k2 == 1:
        blurred2 = img
    elif method == 'Gaussian':
        blurred2 = cv2.GaussianBlur(img, (k2, k2), 0)
    elif method == 'Average':
        blurred2 = cv2.blur(img, (k2, k2))
    difference = cv2.subtract(blurred1, blurred2)
    return difference

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

def make_convex(mask, use_approxPolyDp = True, use_convexHull = True, app_ratio = 0.01, combine_cnts = False):
    contours, hierarchy = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE )
    mask_convex = np.zeros(mask.shape, dtype=np.uint8)
    if combine_cnts:
        cnt_combine = None
        for cnt_idx, cnt in enumerate(contours):
            if hierarchy[0, cnt_idx, 3] != -1:
                continue
            if cnt_combine is None:
                cnt_combine = cnt
            else:
                cnt_combine = np.vstack((cnt_combine, cnt))
        if use_approxPolyDp:
            cnt_combine = cv2.approxPolyDP(cnt_combine, app_ratio * cv2.arcLength(cnt_combine, True), True)
        if use_convexHull:
            cnt_combine = cv2.convexHull(cnt_combine)
        cv2.drawContours(mask_convex, [cnt_combine], 0, 255, -1)
        return mask_convex, cnt_combine
    else:
        cnt = None
        for cnt_idx, cnt in enumerate(contours):
            if hierarchy[0, cnt_idx, 3] != -1:
                continue
            if use_approxPolyDp:
                cnt = cv2.approxPolyDP(cnt, app_ratio * cv2.arcLength(cnt, True), True)
            if use_convexHull:
                cnt = cv2.convexHull(cnt)
            cv2.drawContours(mask_convex, [cnt], 0, 255, -1)
        return mask_convex, cnt

def mask2cnt(mask):
    contours, hierarchy = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE )
    return contours[0]

def find_largest_CC(mask, min_convex_rate = 0, min_area = 0, ref_p = None, max_dist_ref_p = 0):
    '''
    Find largest connected component in a mask image, with minimum @min_convex_rate and @min_area.
    Can also set a reference point @ref_p so that the center of the found connected component is at maximum @max_dist_ref_p to the reference point.
    Return a mask with only the largest connected component drawn, as well as the max contour area.
    Returns (None, -1) if nothing good found.
    '''
    contours, hierarchy = cv2.findContours(mask.copy(), mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE )
    max_area = 0
    max_cnt = None
    for cnt_idx, cnt in enumerate(contours):
        if hierarchy[0, cnt_idx, 3] != -1:
            continue
        if min_area > 0 and cv2.contourArea(cnt) < min_area:
            continue
        if min_convex_rate > 0 and not is_roughly_convex(cnt, threshold = min_convex_rate):
            continue
        if ref_p is not None:
            mean_p = cnt.mean(axis = 0)[0]
            mean_p = mean_p[::-1]
            if euc_dist(mean_p, ref_p) > max_dist_ref_p:
                continue
        cnt_area = cv2.contourArea(cnt)
        if cnt_area > max_area:
            max_area = cnt_area
            max_cnt = cnt
    if max_cnt is None:
        return (None, -1)
    max_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(max_mask, [max_cnt], 0, 255, -1)
    #for cnt_idx, cnt in enumerate(contours):
    #    if hierarchy[0, cnt_idx, 3] != -1:
    #        cv2.drawContours(max_mask, contours, cnt_idx, 0, -1)
    return max_mask, max_area

def get_closest_contour(contours, hierarchy, ref_loc, min_span = 0, min_length = 0, hierarchy_req = None):
    '''
    Get a contour closest to the reference point @ref_loc, with minimal size (span) of @min_span
    @hierarchy_req specifies whether we are interested in only the inner contours, outer contours, or both
    '''
    min_dist = 10000
    closest_cnt = None
    for cnt_idx, cnt in enumerate(contours):
        if hierarchy_req == 'inner' and hierarchy[0, cnt_idx, 3] == -1:
            continue
        elif hierarchy_req == 'outer' and hierarchy[0, cnt_idx, 3] != -1:
            continue
        max_p = cnt.max(axis = 0)
        min_p = cnt.min(axis = 0)
        diff_p = max_p - min_p
        if diff_p.min() > min_span and diff_p.max() > min_length:
            mean_p = cnt.mean(axis = 0)[0]
            mean_loc = mean_p[::-1] # convert from (x, y) to (row_idx, col_idx)
            dist = euc_dist(mean_loc, ref_loc)
            if dist < min_dist:
                min_dist = dist
                closest_cnt = cnt
    return closest_cnt

def get_closest_blob(mask, ref_loc, min_span = 0, min_length = 0, hierarchy_req = None):
    mask_closest = np.zeros(mask.shape, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE)
    closest_cnt = get_closest_contour(contours, hierarchy, ref_loc, min_span, min_length, hierarchy_req)
    if closest_cnt is None:
        return None
    cv2.drawContours(mask_closest, [closest_cnt], 0, 255, -1)
    return mask_closest

def get_contour_center(cnt):
    p_center = cnt.mean(axis = 0)[0].astype(np.float32)
    return p_center

def get_small_blobs(mask, max_peri = None, max_area = None, max_span = None):
    '''
    For a @mask, find all the connected components that are small
    Can set thresholds based on perimeter, area, or span
    Holes are not considered as blobs
    Return the mask with only small blobs, as well as the number of blobs remained
    '''
    mask_small = np.zeros(mask.shape, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE)
    counter = 0
    for cnt_idx, cnt in enumerate(contours):
        if hierarchy[0, cnt_idx, 3] != -1: # not holes
            continue
        if max_peri is not None and len(cnt) > max_peri:
            continue
        if max_area is not None and cv2.contourArea(cnt) > max_area:
            continue
        if max_span is not None:
            max_p = cnt.max(axis = 0)
            min_p = cnt.min(axis = 0)
            diff_p = max_p - min_p
            if diff_p.max() + 1 > max_span:
                continue
        cv2.drawContours(mask_small, contours, cnt_idx, 255, -1)
        counter += 1
    return mask_small, counter

def get_big_blobs(mask, min_peri = None, min_area = None, min_span = None):
    '''
    For a @mask, find all the connected components that are big
    Can set thresholds based on perimeter, area, or span
    Holes are not considered as blobs
    Return the mask with only big blobs, as well as the number of blobs remained
    '''
    mask_big = mask.copy()
    contours, hierarchy = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE)
    counter = 0
    for cnt_idx, cnt in enumerate(contours):
        if hierarchy[0, cnt_idx, 3] != -1: # don't care holes
            continue
        if min_peri is not None and len(cnt) < min_peri:
            cv2.drawContours(mask_big, contours, cnt_idx, 0, -1)
            continue
        if min_area is not None and cv2.contourArea(cnt) < min_area:
            cv2.drawContours(mask_big, contours, cnt_idx, 0, -1)
            continue
        if min_span is not None:
            max_p = cnt.max(axis = 0)
            min_p = cnt.min(axis = 0)
            diff_p = max_p - min_p
            if diff_p.min() + 1 < min_span:
                cv2.drawContours(mask_big, contours, cnt_idx, 0, -1)
                continue
        #cv2.drawContours(mask_big, contours, cnt_idx, 255, -1)
        counter += 1
    return mask_big, counter

def get_square_blobs(mask, th_diff = 0.7, th_area = 0.6):
    mask_square = np.zeros(mask.shape, dtype = np.uint8)
    contours, hierarchy = cv2.findContours(mask, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_NONE)
    counter = 0
    for cnt_idx, cnt in enumerate(contours):
        if is_roughly_square(cnt, th_diff, th_area):
            cv2.drawContours(mask_square, contours, cnt_idx, 255, -1)
            counter += 1
    return mask_square, counter

def expand_with_bound(mask, bound_mask, size = 3):
    mask = cv2.bitwise_and(mask, bound_mask)
    mask = expand(mask, size, method = 'square')
    mask = cv2.bitwise_and(mask, bound_mask)
    return mask

def calc_cumsum(input_array):
    '''
    Calculates cumulative sum of @input_array
    The result is slightly different from numpy's cumsum, so that the sum of block
    with row range [i1, i2] and column range [j1, j2] can be expressed as
    cumsum[i2 + 1, j2 + 1] + cumsum[i1, j1] - cumsum[i1, j2 + 1] - cumsum[i2 + 1, j1]
    '''
    height, width = input_array.shape
    cumsum = np.cumsum(np.cumsum(input_array, axis=0), axis=1)
    new_cumsum = np.zeros((height + 1, width + 1))
    new_cumsum[1:,1:] = cumsum

    return new_cumsum

def skeletonize(mask):
    import skimage.morphology
    skeleton = skimage.morphology.skeletonize(mask > 0)
    return skeleton.astype(np.uint8) * 255

############################### DISPLAY ########################################
def display_image(display_name, img, wait_time = -1, is_resize = True, resize_method = "max", resize_max = -1, resize_scale = 1, save_image = False):
    '''
    Display image at appropriate size. There are two ways to specify the size:
    1. If resize_max is greater than zero, the longer edge (either width or height) of the image is set to this value
    2. If resize_scale is greater than zero, the image is scaled by this factor
    '''
    if is_resize:
        img_shape = img.shape
        height = img_shape[0]; width = img_shape[1]
        if resize_max > 0:
            if height > width:
                img_display = cv2.resize(img, (resize_max * width / height, resize_max), interpolation = cv2.INTER_NEAREST)
            else:
                img_display = cv2.resize(img, (resize_max, resize_max * height / width), interpolation = cv2.INTER_NEAREST)
        elif resize_scale > 0:
            img_display = cv2.resize(img, (width * resize_scale, height * resize_scale), interpolation = cv2.INTER_NEAREST)
        else:
            print "Unexpected parameter in image display. About to exit..."
            sys.exit()
    else:
        img_display = img

    cv2.imshow(display_name, img_display)
    cv2.waitKey(wait_time)
    #if save_image:
    if True:
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        file_path = os.path.join('tmp', display_name + '.png')
        cv2.imwrite(file_path, img_display)

def check_and_display(display_name, img, display_list, wait_time = -1, is_resize = True, resize_method = "max", resize_max = -1, resize_scale = 1, save_image = False):
    if display_name in display_list:
        display_image(display_name, img, wait_time, is_resize, resize_method, resize_max, resize_scale, save_image)

def display_mask(display_name, img, mask, color = (0, 255, 255), wait_time = -1, is_resize = True, resize_method = "max", resize_max = -1, resize_scale = 1, save_image = False):
    img_display = img.copy()
    img_display[mask > 0, :] = color
    display_image(display_name, img_display, wait_time, is_resize, resize_method, resize_max, resize_scale, save_image)

def check_and_display_mask(display_name, img, mask, display_list, color = (0, 255, 255), wait_time = -1, is_resize = True, resize_method = "max", resize_max = -1, resize_scale = 1, save_image = False):
    if display_name in display_list:
        display_mask(display_name, img, mask, color, wait_time, is_resize, resize_method, resize_max, resize_scale, save_image)

def plot_bar(bar_data, name = 'unknown', h = 400, w = 400, wait_time = -1, is_resize = True, resize_method = "max", resize_max = -1, resize_scale = 1, save_image = False):
    n_items = len(bar_data)
    y_max = np.max(bar_data) * 1.1 + 1 # make sure y_max > 0
    plot = np.ones((h, w, 3), dtype = np.uint8) * 255
    for i, bar_h in enumerate(bar_data):
        cv2.rectangle(plot, (int((i + 1 - 0.3) / (n_items + 1) * w), h), (int((i + 1 - 0.3) / (n_items + 1) * w), h - int(bar_h / y_max * h)), [255, 0, 0], -1)
    cv2.putText(plot, "max = %f" % np.max(bar_data), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255])
    display_image(name, plot, wait_time, is_resize, resize_method, resize_max, resize_scale, save_image)

def vis_detections(img, dets, labels, thresh = 0.5):
    # dets format should be: [x1, y1, x2, y2, confidence, cls_idx]
    if len(dets.shape) < 2:
        return img
    inds = np.where(dets[:, -2] >= thresh)[0]

    img_detections = img.copy()
    if len(inds) > 0:
        for i in inds:
            cls_name = labels[int(dets[i, -1] + 0.1)]
            bbox = dets[i, :4]
            score = dets[i, -2]
            text = "%s : %f" % (cls_name, score)
            #print 'Detected roi for %s:%s score:%f' % (cls_name, bbox, score)
            cv2.rectangle(img_detections, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 8)
            cv2.putText(img_detections, text, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img_detections

################################ SHAPE #########################################
def get_border(img, border_width = 5):
    border = np.ones(img.shape[:2], np.bool)
    border[border_width : border.shape[0] - border_width, border_width : border.shape[1] - border_width] = False
    return border

def is_roughly_square(cnt, th_diff = 0.7, th_area = 0.6):
    max_p = cnt.max(axis = 0)
    min_p = cnt.min(axis = 0)
    diff_p = (max_p - min_p)[0]
    return (float(diff_p.min()) / diff_p.max() > th_diff) and (float(cv2.contourArea(cnt)) / diff_p[0] / diff_p[1] > th_area)

def is_roughly_convex(cnt, threshold = 0.9):
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

def is_line_seg_close2(line1, line2):
    pt1_1 = np.array(line1[0 : 2])
    pt1_2 = np.array(line1[2 : 4])
    pt2_1 = np.array(line2[0 : 2])
    pt2_2 = np.array(line2[2 : 4])
    l1 = euc_dist(pt1_1, pt1_2)
    v1 = pt1_2 - pt1_1
    v2 = pt2_1 - pt1_1
    v3 = pt2_2 - pt1_1
    area1 = np.absolute(np.cross(v1, v2))
    area2 = np.absolute(np.cross(v1, v3))
    d1 = area1 * 2 / l1
    d2 = area2 * 2 / l1
    return (d1 <= 3 and d2 <= 3)


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

def sort_pts(pts, order_first = 'y'):
    pts = [tuple(x) for x in pts]
    dtype = [('x', float), ('y', float)]
    pts = np.array(pts, dtype = dtype)
    pts = np.sort(pts, order = order_first)
    pts_rtn = []
    for pt in pts:
        pts_rtn.append(tuple(pt))
    return pts_rtn

def get_corner_pts(bw, perimeter = None, center = None, method = 'line', is_debug = False, sanity_checks = None):
    '''
    Given an input image @bw where the borders of a rough rectangle are masked, the function detects its corners
    Two methods:
    'line' tries to detect four lines first, and
    'point' directly gets the top-left, top-right, bottom-left, bottom-right points
    The function returns @corners as float numbers: [[ul_x, ul_y], [ur_x, ur_y], [bl_x, bl_y], [br_x, br_y]]
    The function returns None if cannot find the corners with confidence
    '''
    if method == 'line':
        center = (center[1], center[0]) # in (x, y) format
        perimeter = int(perimeter)

        lines = cv2.HoughLinesP(bw, 1, np.pi/180, perimeter / 40, minLineLength = perimeter / 20, maxLineGap = perimeter / 20)
        lines = lines[0]

        if is_debug:
            img = np.zeros((bw.shape[0], bw.shape[1], 3), dtype=np.uint8)
            for line in lines:
                pt1 = (line[0], line[1])
                pt2 = (line[2], line[3])
                print (pt1, pt2)
                cv2.line(img, pt1, pt2, (255, 255, 255), 1)
            cv2.namedWindow('test')
            display_image('test', img)

        # get four major lines
        new_lines = list()
        for line in lines:
            flag = True
            for new_line in new_lines:
                if is_line_seg_close(line, new_line):
                    flag = False
                    break
            if flag:
                new_lines.append(list(line))
        if is_debug:
            print "four lines: %s" % new_lines
        if len(new_lines) != 4:
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
                dist = euc_dist(inter_p, center)
                if dist < perimeter / 3:
                    corners.append(inter_p)
        if is_debug:
            print "corners: %s" % corners
        if len(corners) != 4:
            return None

        # put the four corners in order
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
        ul = list(ul)
        ur = list(ur)
        bl = list(bl)
        br = list(br)
        if is_debug:
            print "ul: %s, ur: %s, bl: %s, br: %s" % (ul, ur, bl, br)

        # some sanity check here
        if sanity_checks == "perspective":
            len_b = euc_dist(bl, br)
            len_u = euc_dist(ul, ur)
            len_l = euc_dist(ul, bl)
            len_r = euc_dist(ur, br)
            if len_b < len_u or len_b < len_l or len_b < len_r:
                return None

    elif method == 'point':
        bw = bw.astype(bool)
        row_mtx, col_mtx = np.mgrid[0 : bw.shape[0], 0 : bw.shape[1]]
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

def calc_triangle_area(p1, p2, p3):
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)

def get_rotation_degree(bw):
    lines = cv2.HoughLinesP(bw, 1, np.pi/180, 6, minLineLength = 8, maxLineGap = 5)
    if lines is None:
        return None
    lines = lines[0]
    # plotting lines, for testing only ############################
    #img = np.zeros((bw.shape[0], bw.shape[1], 3), dtype=np.uint8)
    #for line in lines:
    #    pt1 = (line[0], line[1])
    #    pt2 = (line[2], line[3])
    #    cv2.line(img, pt1, pt2, (255, 255, 255), 1)
    #cv2.namedWindow('bw')
    #display_image('bw', bw)
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

def rotate(img, n_iterations = 2):
    '''
    Assuming major line patterns in an image are vertical and horizontal, this function tries to
    correct the rotaion to make vertical lines really vertical and horizontal lines really horizontal.
    '''
    img_ret = img
    rotation_degree = 0
    rotation_mtx = None
    for iteration in xrange(n_iterations): # Sometimes need multiple iterations to get the rotation right
        bw = cv2.cvtColor(img_ret, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(bw, 50, 100)
        rotation_degree_tmp = get_rotation_degree(edges)
        if rotation_degree_tmp is None:
            rtn_msg = {'status' : 'fail', 'message' : 'Cannot get rotation degree'}
            return (rtn_msg, None)
        weight = 1
        for i in xrange(3):
            bw[:] = img_ret[:,:,i][:]
            edges = cv2.Canny(bw, 50, 100)
            d = get_rotation_degree(edges)
            if d is not None:
                rotation_degree_tmp += d
                weight += 1
        rotation_degree_tmp /= weight
        rotation_degree += rotation_degree_tmp
        #print rotation_degree
        img_shape = img.shape
        M = cv2.getRotationMatrix2D((img_shape[1]/2, img_shape[0]/2), rotation_degree, scale = 1)
        rotation_mtx = M
        img_ret = cv2.warpAffine(img, M, (img_shape[1], img_shape[0]))

    rtn_msg = {'status' : 'success'}
    return (rtn_msg, (img_ret, rotation_degree, rotation_mtx))

def crop(img, borders):
    shape = img.shape
    is_color = (len(shape) == 3 and shape[2] > 1)
    if borders is None:
        if is_color:
            bw = get_mask(img)
        else:
            bw = img
        rows, cols = np.nonzero(bw)
        min_row = min(rows); max_row = max(rows)
        min_col = min(cols); max_col = max(cols)
    else:
        min_row, max_row, min_col, max_col = borders
    if is_color:
        img_cropped = img[min_row : max_row + 1, min_col : max_col + 1, :]
    else:
        img_cropped = img[min_row : max_row + 1, min_col : max_col + 1]
    return img_cropped, (min_row, max_row, min_col, max_col)

#def smart_crop(img):
#    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    ret, bi = cv2.threshold(bw, 0, 1, cv2.THRESH_BINARY)
#    # TODO: has a risk that the sum here may excede uint8...
#    sum_0 = bi.sum(axis = 0)
#    sum_1 = bi.sum(axis = 1)
#    i_start = 0; i_end = bi.shape[0] - 1; j_start = 0; j_end = bi.shape[1] - 1
#    i_start_cmp_val = sum_1[int(round(config.BRICK_HEIGHT / 4.0 * 2))] * 0.6
#    while sum_1[i_start] < i_start_cmp_val:
#        i_start += 1
#    i_end_cmp_val = sum_1[bi.shape[0] - 1 - int(round(config.BRICK_HEIGHT / 4.0 * 2))] * 0.6
#    while sum_1[i_end] < i_end_cmp_val:
#        i_end -= 1
#    j_start_cmp_val = sum_0[int(round(config.BRICK_WIDTH / 4.0 * 2))] * 0.6
#    while sum_0[j_start] < j_start_cmp_val:
#        j_start += 1
#    j_end_cmp_val = sum_0[bi.shape[1] - 1 - int(round(config.BRICK_WIDTH / 4.0 * 2))] * 0.6
#    while sum_0[j_end] < j_end_cmp_val:
#        j_end -= 1
#
#    #print (bi.shape, i_start, i_end, j_start, j_end)
#    return img[i_start : i_end + 1, j_start : j_end + 1, :], (i_start, i_end, j_start, j_end)

################################ COLOR #########################################
def normalize_brightness(img, mask = None, method = 'hist', max_percentile = 100, min_percentile = 0):
    shape = img.shape
    if mask is None:
        mask = np.ones((shape[0], shape[1]), dtype=bool)
    if mask.dtype != np.bool:
        mask = mask.astype(bool)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    if method == 'hist':
        hist,bins = np.histogram(v.flatten(),256,[0,256])
        hist[0] = 0
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        v_ret = cdf[v]

    elif method == 'max':
        max_v = np.percentile(v[mask], max_percentile)
        min_v = np.percentile(v[mask], min_percentile)
        v[np.bitwise_and((v < min_v), mask)] = min_v
        # What the hell is converScaleAbs doing??? why need abs???
        v_ret = cv2.convertScaleAbs(v, alpha = 254.0 / (max_v - min_v), beta = -(min_v * 254.0 / (max_v - min_v) - 2))
        #v_ret = v_ret[:,:,0]
        v[mask] = v_ret[mask]
        v_ret = v

    hsv[:,:,2] = v_ret
    img_ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img_ret

def normalize_color(img, mask_info = None, mask_apply = None, method = 'hist', max_percentile = 100, min_percentile = 0):
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
    if method == 'hist': # doesn't work well for over-exposed images
        for i in xrange(3):
            v = img[:,:,i]
            hist,bins = np.histogram(v[mask_info].flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')
            v[mask_apply] = cdf[v[mask_apply]]
            img_ret[:,:,i] = v

    elif method == 'grey':
        img = img.astype(float)
        max_rgb = 0
        for i in xrange(3):
            v = img[:,:,i]
            #print v[mask_info].mean()
            v[mask_apply] = v[mask_apply] / v[mask_info].mean()
            img[:,:,i] = v
            if v[mask_apply].max() > max_rgb:
                max_rgb = v[mask_apply].max()

        img[mask_apply, :] = img[mask_apply, :] * 255 / max_rgb
        img = img.astype(np.uint8)
        img_ret = img

    elif method == 'select_grey':
        img = img.astype(np.int64)
        mask_blue_over_exposed = (img[:,:,0] >= 250)
        mask_green_over_exposed = (img[:,:,1] >= 250)
        mask_red_over_exposed = (img[:,:,2] >= 250)
        #print "Blue over exposure: %d" % mask_blue_over_exposed.sum()
        mask_over_bright = ((img[:,:,0] + img[:,:,1] + img[:,:,2]) >= 666)
        mask_over_exposed = np.bitwise_and(super_bitwise_or((mask_blue_over_exposed, mask_green_over_exposed, mask_red_over_exposed)), mask_over_bright)
        #print "Over exposure: %d" % mask_over_bright.sum()
        mask_info = np.bitwise_and(mask_info, np.invert(mask_over_exposed))

        img = img.astype(float)
        max_rgb = 0
        for i in xrange(3):
            v = img[:,:,i]
            v[mask_apply] = v[mask_apply] / v[mask_info].mean()
            img[:,:,i] = v
            if v[mask_apply].max() > max_rgb:
                max_rgb = v[mask_apply].max()

        img[mask_apply, :] = img[mask_apply, :] * 255 / max_rgb
        img = img.astype(np.uint8)
        img = normalize_brightness(img, mask = mask_apply, max_percentile = 90, method = 'max')
        img[mask_over_exposed, 0] = 255
        img[mask_over_exposed, 1] = 255
        img[mask_over_exposed, 2] = 255
        img_ret = img

    elif method == 'max':
        #b, g, r = cv2.split(img)
        #img = cv2.merge((b, g, r))
        for i in xrange(3):
            v = img[:,:,i]
            max_v = np.percentile(v[mask], max_percentile)
            min_v = np.percentile(v[mask], min_percentile)
            v[v < min_v] = min_v
            v_ret = cv2.convertScaleAbs(v, alpha = 220.0 / (max_v - min_v), beta = -(min_v * 220.0 / (max_v - min_v) - 35))
            v_ret = v_ret[:,:,0]
            v[mask] = v_ret[mask]
            img_ret[:,:,i] = v

    return img_ret

def color_inrange(img, color_space, hsv = None, B_L = 0, B_U = 255, G_L = 0, G_U = 255, R_L = 0, R_U = 255,
                                                H_L = 0, H_U = 179, S_L = 0, S_U = 255, V_L = 0, V_U = 255,
                                                L = 0, U = 255):
    if color_space == 'BGR':
        lower_range = np.array([B_L, G_L, R_L], dtype=np.uint8)
        upper_range = np.array([B_U, G_U, R_U], dtype=np.uint8)
        mask = cv2.inRange(img, lower_range, upper_range)
    elif color_space == 'HSV':
        if hsv is None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if H_L <= H_U:
            lower_range = np.array([H_L, S_L, V_L], dtype=np.uint8)
            upper_range = np.array([H_U, S_U, V_U], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_range, upper_range)
        else:
            lower_range1 = np.array([H_L, S_L, V_L], dtype=np.uint8)
            upper_range1 = np.array([180, S_U, V_U], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_range1, upper_range1)
            lower_range2 = np.array([0, S_L, V_L], dtype=np.uint8)
            upper_range2 = np.array([H_U, S_U, V_U], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_range2, upper_range2)
            mask = np.bitwise_or(mask1, mask2)
    elif color_space == 'single':
        lower_range = np.array([L], dtype=np.uint8)
        upper_range = np.array([U], dtype=np.uint8)
        mask = cv2.inRange(img, lower_range, upper_range)

    return mask

def color_dist(img, color_space, hsv = None, BGR_ref = [255, 255, 255], HSV_ref = [0, 255, 255], useV = False):
    if color_space == 'BGR':
        dist = np.sum((img - np.array(BGR_ref, dtype=np.int)), axis = 2)
    elif color_space == 'HSV':
        if hsv is None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        if useV:
            v = hsv[:, :, 2]
            dist = np.absolute(np.cos((h.astype(np.int) - HSV_ref[0]) / 180 * np.pi)) * s * np.sqrt((255 - np.abs(v - HSV_ref[2])) / 255)
        else:
            dist = np.absolute(np.cos((h.astype(np.int) - HSV_ref[0]) / 180 * np.pi)) * s
        dist = 255 - dist
        dist = dist.astype(np.uint8)

    return dist

def detect_color(img_hsv, color, on_surface = False):
    '''
    detect the area in @img_hsv with a specific @color, and return the @mask
    @img_hsv is the input in HSV color space
    @color is a string, describing color
    Currently supported colors: Black, White
    In OpenCV HSV space, H is in [0, 179], the other two are in [0, 255]
    '''
    if color == "black":
        mask1_1 = color_inrange(None, 'HSV', hsv = img_hsv[0], V_U = 50)
        mask1_2 = color_inrange(None, 'HSV', hsv = img_hsv[1], S_U = 60)
        mask1 = cv2.bitwise_and(mask1_1, mask1_2)
        mask2_1 = color_inrange(None, 'HSV', hsv = img_hsv[0], V_U = 20)
        mask2_2 = color_inrange(None, 'HSV', hsv = img_hsv[1], S_U = 100)
        mask2 = cv2.bitwise_and(mask2_1, mask2_2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "white":
        mask = color_inrange(None, 'HSV', hsv = img_hsv, S_U = 60, V_L = 190)
    else:
        print "ERROR: color detection has specified an undefined color!!!!"

    return mask


def detect_colors(img, mask_src, on_surface = False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mask_src is None:
        mask_src = np.ones(img.shape[0:2], dtype = np.uint8) * 255
    mask_nothing = np.zeros(mask_src.shape, dtype = np.uint8)
    # detect green
    mask_green = color_inrange(img, 'HSV', hsv = hsv, H_L = 45, H_U = 96, S_L = 80)
    mask_green = cv2.bitwise_and(mask_green, mask_src)
    mask_green_bool = mask_green.astype(bool)
    if np.any(mask_green_bool) and has_a_brick(mask_green, min_area = 20, min_span = 5):
        S_mean = np.median(hsv[mask_green_bool, 1])
        mask_green = color_inrange(img, 'HSV', hsv = hsv, H_L = 45, H_U = 96, S_L = int(S_mean * 0.7))
        if not has_a_brick(cv2.bitwise_and(mask_green, mask_src), min_area = 20, min_span = 5):
            mask_green = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_green_bool, 2], 75)
            mask_green_on = color_inrange(img, 'HSV', hsv = hsv, H_L = 45, H_U = 96, S_L = int(S_mean * 0.7), V_L = V_ref * 0.75)
            mask_green = (mask_green, mask_green_on)
    else:
        mask_green = mask_nothing if not on_surface else (mask_nothing, mask_nothing)
    # detect yellow
    mask_yellow = color_inrange(img, 'HSV', hsv = hsv, H_L = 8, H_U = 45, S_L = 90)
    mask_yellow = cv2.bitwise_and(mask_yellow, mask_src)
    mask_yellow_bool = mask_yellow.astype(bool)
    if np.any(mask_yellow_bool) and has_a_brick(mask_yellow, min_area = 20, min_span = 5):
        S_mean = np.median(hsv[mask_yellow_bool, 1])
        mask_yellow = color_inrange(img, 'HSV', hsv = hsv, H_L = 8, H_U = 45, S_L = int(S_mean * 0.7))
        if not has_a_brick(cv2.bitwise_and(mask_yellow, mask_src), min_area = 20, min_span = 5):
            mask_yellow = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_yellow_bool, 2], 75)
            mask_yellow_on = color_inrange(img, 'HSV', hsv = hsv, H_L = 8, H_U = 45, S_L = int(S_mean * 0.7), V_L = V_ref * 0.75)
            mask_yellow = (mask_yellow, mask_yellow_on)
    else:
        mask_yellow = mask_nothing if not on_surface else (mask_nothing, mask_nothing)
    # detect red
    mask_red1 = color_inrange(img, 'HSV', hsv = hsv, H_L = 0, H_U = 10, S_L = 105)
    mask_red2 = color_inrange(img, 'HSV', hsv = hsv, H_L = 160, H_U = 179, S_L = 105)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.bitwise_and(mask_red, mask_src)
    mask_red_bool = mask_red.astype(bool)
    if np.any(mask_red_bool) and has_a_brick(mask_red, min_area = 20, min_span = 5):
        S_mean = np.median(hsv[mask_red_bool, 1])
        mask_red1 = color_inrange(img, 'HSV', hsv = hsv, H_L = 0, H_U = 10, S_L = int(S_mean * 0.7))
        mask_red2 = color_inrange(img, 'HSV', hsv = hsv, H_L = 160, H_U = 179, S_L = int(S_mean * 0.7))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        if not has_a_brick(cv2.bitwise_and(mask_red, mask_src), min_area = 20, min_span = 5):
            mask_red = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_red_bool, 2], 75)
            mask_red1_on = color_inrange(img, 'HSV', hsv = hsv, H_L = 0, H_U = 10, S_L = int(S_mean * 0.7), V_L = V_ref * 0.75)
            mask_red2_on = color_inrange(img, 'HSV', hsv = hsv, H_L = 160, H_U = 179, S_L = int(S_mean * 0.7), V_L = V_ref * 0.75)
            mask_red_on = cv2.bitwise_or(mask_red1_on, mask_red2_on)
            mask_red = (mask_red, mask_red_on)
    else:
        mask_red = mask_nothing if not on_surface else (mask_nothing, mask_nothing)
    # detect blue
    mask_blue = color_inrange(img, 'HSV', hsv = hsv, H_L = 93, H_U = 140, S_L = 125)
    mask_blue = cv2.bitwise_and(mask_blue, mask_src)
    mask_blue_bool = mask_blue.astype(bool)
    if np.any(mask_blue_bool) and has_a_brick(mask_blue, min_area = 20, min_span = 5):
        S_mean = np.median(hsv[mask_blue_bool, 1])
        mask_blue = color_inrange(img, 'HSV', hsv = hsv, H_L = 93, H_U = 140, S_L = int(S_mean * 0.8))
        if not has_a_brick(cv2.bitwise_and(mask_blue, mask_src), min_area = 20, min_span = 5):
            mask_blue = mask_nothing
        if on_surface:
            V_ref = np.percentile(hsv[mask_blue_bool, 2], 75)
            mask_blue_on = color_inrange(img, 'HSV', hsv = hsv, H_L = 93, H_U = 140, S_L = int(S_mean * 0.8), V_L = V_ref * 0.75)
            mask_blue = (mask_blue, mask_blue_on)
    else:
        mask_blue = mask_nothing if not on_surface else (mask_nothing, mask_nothing)

    return (mask_green, mask_red, mask_yellow, mask_blue)

def detect_colorful(img, on_surface = False):
    lower_bound = [0, 100, 20]
    upper_bound = [179, 255, 255]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = np.array(lower_bound, dtype=np.uint8)
    upper_range = np.array(upper_bound, dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    return mask

def checkBlurByGradient(img, gradientPatchNBox = 5, gradientPatchWidth = 25, gradientPatchHeight = 25, threshold = 500):
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n_rows, n_cols = img.shape[:2]
    max_gradients = 0
    for i in xrange(gradientPatchNBox):
        for j in xrange(gradientPatchNBox):
            top = (n_rows / ( 2 * gradientPatchNBox + 1)) * (2 * i + 1);
            left = (n_cols / ( 2 * gradientPatchNBox + 1)) * (2 * j + 1);
            bw_window = bw[top : top + gradientPatchHeight, left : left + gradientPatchWidth]
            gradients = np.absolute(cv2.Sobel(bw_window, cv2.CV_64F, 1, 1, ksize = 5))
            sum_gradients = np.sum(gradients)
            if sum_gradients > max_gradients:
                max_gradients = sum_gradients
    #print max_gradients
    if max_gradients > threshold:
        return False
    else:
        return True

########################## OBJECT DETECTION ###################################
### http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the ratio of overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that overlap too much
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), pick

def cv_img2sk_img(cv_img):
    sk_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return sk_img

def object_detection(cv_img, labels, window_method = "region", recognition_method = "cnn", net = None, transformer = None):
    import dlib

    ## selective search to find candidate bounding boxes
    sk_img = cv_img2sk_img(cv_img)
    rects = [] # locations of the candidates
    dlib.find_candidate_object_locations(sk_img, rects, min_size = cv_img.shape[0] * cv_img.shape[1] / 100)

    ## count how many rects are promissing
    rect_candidates = []
    for rect in rects:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        aspect_ratio = float(y2 - y1) / (x2 - x1)
        if aspect_ratio > 0.6 and aspect_ratio < 1.6 \
                and y1 != 0 and y2 != sk_img.shape[0] - 1 and x1 != 0 and x2 != sk_img.shape[1] - 1:
            rect_candidates.append([x1, y1, x2, y2])
            if len(rect_candidates) > 50:
                break
    print "# of candidate rectangles: %d" % len(rect_candidates)

    ## if nothing interesting in the image
    if not rect_candidates:
        return ([], [])

    ## do recognition for the rects in a batch
    data_layer = net.blobs['data']
    data_layer.reshape(len(rect_candidates), data_layer.channels, data_layer.height, data_layer.width)
    for idx, rect in enumerate(rect_candidates):
        x1, y1, x2, y2 = rect
        img_rect = cv_img[y1 : y2 + 1, x1 : x2 + 1, :]
        net.blobs['data'].data[idx, ...] = transformer.preprocess('data', img_rect)

    ## real DNN processing
    out = net.forward()
    label_idxes = out['prob'].argmax(axis = 1)

    ## find rects that are not just background
    rect_detected = []
    label_idxes_detected = []
    for idx, rect in enumerate(rect_candidates):
        if label_idxes[idx] != len(labels) - 1:
            rect_detected.append(rect)
            label_idxes_detected.append(label_idxes[idx])

    return (rect_detected, label_idxes_detected)

def object_recognition(cv_img, recognition_method = "cnn", svm_classifiers = [], net = None, transformer = None, hog_svm = None, hog_descriptor = None):

    if recognition_method == "cnn":
        ## set up input later of net
        data_layer = net.blobs['data']
        data_layer.reshape(1, data_layer.channels, data_layer.height, data_layer.width)
        data_layer.data[0, ...] = transformer.preprocess('data', cv_img)

        ## real DNN processing
        out = net.forward()
        label_idx = out['prob'].argmax()
    elif recognition_method == "hog-svm":
        bw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        if bw.shape[0] != 128 or bw.shape[1] != 128:
            bw = cv2.resize(bw, (128, 128))
        hog_features = np.transpose(hog_descriptor.compute(bw))
        label_idx = int(hog_svm.predict(hog_features))

    elif recognition_method == "s-svm":
        sk_img = cv_img2sk_img(cv_img)
        label_idx = -1
        for idx, detector in enumerate(svm_classifiers):
            dets = detector(sk_img)
            if len(dets) > 0:
                if label_idx == -1:
                    label_idx = idx
                else:
                    label_idx = -1
                    break
        if label_idx == -1:
            label_idx = len(svm_classifiers)

    return label_idx

################### FACE DETECTION & RECOGNITION ##############################
def detect_face(img, method = "openface", openface_align = None):
    if method == "openface":
        bb = openface_align.getLargestFaceBoundingBox(img)
        return bb

def get_face_feature(img, align, align_img_dim, net):
    bb = align.getLargestFaceBoundingBox(img)
    if bb is None:
        return None

    alignedFace = align.alignImg("affine", align_img_dim, img, bb)
    if alignedFace is None:
        return None

    rep = net.forwardImage(alignedFace)

    return rep

