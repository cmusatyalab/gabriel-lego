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

DISPLAY_LIST = ['input', 'board', 'board_edge', 'board_corrected', 'lego', 'lego_perspective', 'lego_edge', 'lego_correct', 'lego_cropped', 'lego_syn']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help = "The image to process",
                       )
    args = parser.parse_args()
    return args.input_file


input_file = parse_arguments()
img = cv2.imread(input_file)
cv2.namedWindow("input")
lc.display_image("input", img)

for display_name in DISPLAY_LIST:
    cv2.namedWindow(display_name)

rtn_msg, img_lego, perspective_mtx = lc.locate_lego(img, DISPLAY_LIST)
print rtn_msg
rtn_msg, img_lego_correct = lc.correct_orientation(img_lego, perspective_mtx, DISPLAY_LIST)
print rtn_msg
rtn_msg, bitmap = lc.reconstruct_lego(img_lego_correct, DISPLAY_LIST)
print rtn_msg
img_syn = lc.bitmap2syn_img(bitmap)
lc.display_image('lego_syn', img_syn)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
