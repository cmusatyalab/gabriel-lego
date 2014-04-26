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

img = cv2.imread("test_images_all/frame-408.jpeg")
cv2.namedWindow("original")
lc.display_image("original", img)

#cv2.namedWindow("black")
cv2.namedWindow("board")
cv2.namedWindow("edge")
cv2.namedWindow("lego")
#cv2.namedWindow("test")

img = lc.locate_lego(img)

#cv2.floodFill(mask_black, None, tuple(contour[0, 0]), 0)
#cv2.drawContours(img, contours, -1, (0,255,0), -1)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.stdout.write("user exits\n")
