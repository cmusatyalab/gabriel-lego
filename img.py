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
# This script is used for testing computer vision algorithms in the 
# Lego Task Assistance project. It does processing for one image.
# Usage: python img.py <image-path>
#

'''
This script loads a single image from file, and try to generate a Lego symbolic representation out of it.
It is primarily used as a quick test tool for the computer vision algorithm.
'''

import cv2
import sys
import time
import argparse
import lego_cv as lc
import bitmap as bm
import lego_config as config

config.setup(is_streaming = False)
lc.set_config(is_streaming = False)
display_list = config.DISPLAY_LIST_TEST

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help = "The image to process",
                       )
    args = parser.parse_args()
    return args.input_file


input_file = parse_arguments()
img = cv2.imread(input_file)
if img.shape != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3):
    img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

for display_name in display_list:
    cv2.namedWindow(display_name)
if 'input' in display_list:
    lc.display_image("input", img, wait_time = config.DISPLAY_WAIT_TIME, resize_max = config.DISPLAY_MAX_PIXEL, save_image = config.SAVE_IMAGE)

rtn_msg, objects = lc.find_lego(img, display_list)
if objects is not None:
    img_lego, img_lego_full, img_board, img_board_ns, perspective_mtx = objects
print rtn_msg
if rtn_msg['status'] == 'success':
    rtn_msg, objects = lc.correct_orientation(img_lego, img_lego_full, display_list)
    if objects is not None:
        img_lego_correct, img_lego_full_correct, rotation_mtx = objects
    print rtn_msg
if rtn_msg['status'] == 'success':
    rtn_msg, bitmap = lc.reconstruct_lego(img_lego, img_board, img_board_ns, rotation_mtx, display_list)
    print rtn_msg
    if rtn_msg['status'] == 'success':
        img_syn = bm.bitmap2syn_img(bitmap)
        lc.check_and_display('lego_syn', img_syn, display_list)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
