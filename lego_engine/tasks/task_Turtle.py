#!/usr/bin/env python
import numpy as np

# Labels: nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6,
# unsure:7
bitmaps = [np.array([[2, 2, 2, 2]]),
           np.array([[0, 0, 1, 0],
                     [2, 2, 2, 2]]),
           np.array([[0, 0, 1, 6],
                     [2, 2, 2, 2]]),
           np.array([[0, 0, 1, 1],
                     [0, 0, 1, 6],
                     [2, 2, 2, 2]]),
           np.array([[0, 0, 1, 1],
                     [0, 2, 1, 6],
                     [2, 2, 2, 2]]),
           np.array([[0, 2, 1, 1],
                     [0, 2, 1, 6],
                     [2, 2, 2, 2]]),
           np.array([[0, 2, 2, 2],
                     [0, 2, 1, 1],
                     [0, 2, 1, 6],
                     [2, 2, 2, 2]])
           ]

# time_estimates = [7.46666667, 11.76666667, 11.93333333, 9.03333333, 13.4,
# 12.03333333, 10.63333333, 15.16666667, 10.93333333]
time_estimates = [6.1, 11.26666667, 9.86666667, 9.33333333, 11.3, 12.03333333,
                  10.83333333]
