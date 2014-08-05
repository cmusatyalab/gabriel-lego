#!/usr/bin/env python
import numpy as np

# Labels: nothing:0, white:1, green:2, yellow:3, red:4, blue:5, black:6, unsure:7
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
                     [2, 2, 2, 2]]),
           ]


