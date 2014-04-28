#!/usr/bin/env python
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0

import cv2
import sys
import time
import math
import numpy as np
import lego_cv as lc

DISPLAY_LIST = ['input', 'board', 'board_corrected', 'lego', 'lego_perspective', 'lego_edge', 'lego_correct', 'lego_cropped', 'lego_syn']

img = cv2.imread("test_images2/frame-03302.jpeg")
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
