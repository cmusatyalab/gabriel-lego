#!/usr/bin/env python
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0

import cv2
import sys
import time
import math
import argparse
import numpy as np
import lego_cv as lc
import lego_config as config

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

for display_name in display_list:
    cv2.namedWindow(display_name)
if 'input' in display_list:
    lc.display_image("input", img)

rtn_msg, img_lego, perspective_mtx = lc.locate_lego(img, display_list)
print rtn_msg
rtn_msg, img_lego_correct = lc.correct_orientation(img_lego, perspective_mtx, display_list)
print rtn_msg
rtn_msg, bitmap = lc.reconstruct_lego(img_lego_correct, display_list)
print rtn_msg
img_syn = lc.bitmap2syn_img(bitmap)
lc.display_image('lego_syn', img_syn)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
