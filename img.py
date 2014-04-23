#!/usr/bin/env python
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0

import cv2
import sys
import time
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
lc.display_image("black", img_black)


try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
