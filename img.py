#!/usr/bin/env python
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0

import cv2
import sys
import time
import math
import lego_cv as lc
import numpy as np

img = cv2.imread("test_images/frame-408.jpeg")
cv2.namedWindow("original")
lc.display_image("original", img)

black_min = np.array([0, 0, 0], np.uint8)
black_max = np.array([100, 100, 100], np.uint8)
mask_black = cv2.inRange(img, black_min, black_max)
img_black = cv2.bitwise_and(img, img, mask = mask_black)
cv2.namedWindow("black")
lc.display_image("black", mask_black)

mask_black_copy = mask_black.copy()
contours, hierarchy = cv2.findContours(mask_black_copy, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_NONE )
mask_black_dots = mask_black.copy()
counts = np.zeros((9, 16))
for contour in contours:
    contour.resize(len(contour), 2)
    if len(contour) > 20:
        lc.set_value(mask_black_dots, contour, 0)
        #lc.display_image("black", mask_black_dots)
        continue
    max_p = contour.max(axis = 0)
    min_p = contour.min(axis = 0)
    #print "max: %s, min: %s" % (max_p, min_p)
    diff_p = max_p - min_p
    if diff_p.max() > 5:
        lc.set_value(mask_black_dots, contour, 0)
        continue
    mean_p = contour.mean(axis = 0)
    counts[int(mean_p[1] / 80), int(mean_p[0] / 80)] += 1

cv2.namedWindow("black_dots")
cv2.drawContours(img, contours, -1, (0,255,0), 3)
lc.display_image("black_dots", img)

print counts

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
