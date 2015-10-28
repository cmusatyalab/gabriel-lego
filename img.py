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

import argparse
import cv2
import sys
import time

sys.path.insert(0, "..")
import config
import bitmap as bm
import lego_cv as lc
import zhuocv as zc

config.setup(is_streaming = False)
lc.set_config(is_streaming = False)
display_list = config.DISPLAY_LIST

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help = "The image to process",
                       )
    args = parser.parse_args()
    return args.input_file


# load test image
input_file = parse_arguments()
img = cv2.imread(input_file)
stretch_ratio = float(16) / 9 * img.shape[0] / img.shape[1]
if img.shape != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3):
    img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)

zc.check_and_display("input", img, display_list, resize_max = config.DISPLAY_MAX_PIXEL, wait_time = config.DISPLAY_WAIT_TIME)

# process image and get the symbolic representation
rtn_msg, objects = lc.process(img, stretch_ratio, display_list)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
