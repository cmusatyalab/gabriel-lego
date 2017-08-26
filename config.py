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

# If True, configurations are set to process video stream in real-time (use with lego_server.py)
# If False, configurations are set to process one independent image (use with img.py)
IS_STREAMING = True

RECOGNIZE_ONLY = False

# Port for communication between proxy and task server
TASK_SERVER_PORT = 6090

# Port for communication between master and workder proxies
MASTER_SERVER_PORT = 6091

# Whether or not to save the displayed image in a temporary directory
SAVE_IMAGE = False

# Convert all incoming frames to a fixed size to ease processing
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640

BLUR_KERNEL_SIZE = IMAGE_WIDTH / 16 + 1

# Display
DISPLAY_MAX_PIXEL = 640
DISPLAY_SCALE = 5
DISPLAY_LIST_ALL = ['test', 'input', 'DoB', 'mask_black', 'mask_black_dots',
                    'board', 'board_border_line', 'board_edge', 'board_grey', 'board_mask_black', 'board_mask_black_dots', 'board_DoB', 'edge_inv',
                    'edge',
                    'board_n0', 'board_n1', 'board_n2', 'board_n3', 'board_n4', 'board_n5', 'board_n6',
                    'lego_u_edge_S', 'lego_u_edge_norm_L', 'lego_u_dots_L', 'lego_full', 'lego', 'lego_only_color',
                    'lego_correct', 'lego_rect', 'lego_cropped', 'lego_color', 'plot_line', 'lego_syn',
                    'guidance']
DISPLAY_LIST_TEST = ['input', 'board', 'lego_u_edge_S', 'lego_u_edge_norm_L', 'lego_u_dots_L', 'lego_syn']
DISPLAY_LIST_STREAM = ['input', 'lego_syn']
#DISPLAY_LIST_TASK = ['input', 'board', 'lego_syn', 'guidance']
DISPLAY_LIST_TASK = []
if not IS_STREAMING:
    DISPLAY_LIST = DISPLAY_LIST_TEST
else:
    if RECOGNIZE_ONLY:
        DISPLAY_LIST = DISPLAY_LIST_STREAM
    else:
        DISPLAY_LIST = DISPLAY_LIST_TASK
DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500

## Black dots
BD_COUNT_N_ROW = 9
BD_COUNT_N_COL = 16
BD_BLOCK_HEIGHT = IMAGE_HEIGHT / BD_COUNT_N_ROW
BD_BLOCK_WIDTH = IMAGE_WIDTH / BD_COUNT_N_COL
BD_BLOCK_SPAN = max(BD_BLOCK_HEIGHT, BD_BLOCK_WIDTH)
BD_BLOCK_AREA = BD_BLOCK_HEIGHT * BD_BLOCK_WIDTH
BD_COUNT_THRESH = 30
BD_MAX_PERI = (IMAGE_HEIGHT + IMAGE_HEIGHT) / 40
BD_MAX_SPAN = int(BD_MAX_PERI / 4.0 + 0.5)
# Two ways to check black dot size:
# 'simple': check contour length and area
# 'complete": check x & y max span also
CHECK_BD_SIZE = 'simple'

## Color detection
# H: hue, S: saturation, V: value (which means brightness)
# L: lower_bound, U: upper_bound, TH: threshold
# TODO:
BLUE = {'H' : 110, 'S_L' : 100, 'B_TH' : 110} # H: 108
YELLOW = {'H' : 30, 'S_L' : 100, 'B_TH' : 170} # H: 25 B_TH: 180
GREEN = {'H' : 70, 'S_L' : 100, 'B_TH' : 60} # H: 80 B_TH: 75
RED = {'H' : 0, 'S_L' : 100, 'B_TH' : 130}
BLACK = {'S_U' : 70, 'V_U' : 60}
#WHITE = {'S_U' : 60, 'B_L' : 101, 'B_TH' : 160} # this includes side white, too
WHITE = {'S_U' : 60, 'V_L' : 150}
BLACK_DOB_MIN_V = 15
BD_DOB_MIN_V = 30
# If using a labels to represent color, this is the right color: 0 means nothing (background) and 7 means unsure
COLOR_ORDER = ['nothing', 'white', 'green', 'yellow', 'red', 'blue', 'black', 'unsure']

## Board
BOARD_MIN_AREA = BD_BLOCK_AREA * 7
BOARD_MIN_LINE_LENGTH = BD_BLOCK_SPAN
BOARD_MIN_VOTE = BD_BLOCK_SPAN / 2
# Once board is detected, convert it to a perspective-corrected standard size for further processing
BOARD_RECONSTRUCT_HEIGHT = 155 * 1
BOARD_RECONSTRUCT_WIDTH = 270 * 1
BOARD_BD_MAX_PERI = (BOARD_RECONSTRUCT_HEIGHT + BOARD_RECONSTRUCT_WIDTH) / 30
BOARD_BD_MAX_SPAN = int(BOARD_BD_MAX_PERI / 4.0 + 1.5)
BOARD_RECONSTRUCT_AREA = BOARD_RECONSTRUCT_HEIGHT * BOARD_RECONSTRUCT_WIDTH
BOARD_RECONSTRUCT_PERI = (BOARD_RECONSTRUCT_HEIGHT + BOARD_RECONSTRUCT_WIDTH) * 2
BOARD_RECONSTRUCT_CENTER = (BOARD_RECONSTRUCT_HEIGHT / 2, BOARD_RECONSTRUCT_WIDTH / 2)

## Bricks
BRICK_HEIGHT = BOARD_RECONSTRUCT_HEIGHT / 12.25 # magic number
BRICK_WIDTH = BOARD_RECONSTRUCT_WIDTH / 26.2 # magic number
BRICK_HEIGHT_THICKNESS_RATIO = 15 / 12.25 # magic number
BLOCK_DETECTION_OFFSET = 2

## Optimizations
# If True, performs a second step fine-grained board detection algorithm.
# Depending on the other algorithms, this is usually not needed.
OPT_FINE_BOARD = False

# Treat background pixels differently
OPT_NOTHING = False

BM_WINDOW_MIN_TIME = 0.1
BM_WINDOW_MIN_COUNT = 1

# The percentage of right pixels in each block must be higher than this threshold
WORST_RATIO_BLOCK_THRESH = 0.6

# If True, do perspective correction first, then color normalization
# If False, do perspective correction after color has been normalized
# Not used anymore...
PERS_NORM = True

## Consts
ACTION_ADD = 0
ACTION_REMOVE = 1
ACTION_TARGET = 2
ACTION_MOVE = 3

DIRECTION_NONE = 0
DIRECTION_UP = 1
DIRECTION_DOWN = 2

GOOD_WORDS = ["Excellent. ", "Great. ", "Good job. ", "Wonderful. "]

def setup(is_streaming):
    global IS_STREAMING, DISPLAY_LIST, DISPLAY_WAIT_TIME, SAVE_IMAGE
    IS_STREAMING = is_streaming
    if not IS_STREAMING:
        DISPLAY_LIST = DISPLAY_LIST_TEST
    else:
        if RECOGNIZE_ONLY:
            DISPLAY_LIST = DISPLAY_LIST_STREAM
        else:
            DISPLAY_LIST = DISPLAY_LIST_TASK
    DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500
    SAVE_IMAGE = not IS_STREAMING

