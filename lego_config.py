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

IS_STREAMING = True
SAVE_IMAGE = False

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
BLUR_KERNEL_SIZE = IMAGE_WIDTH / 16 + 1

# Display
DISPLAY_MAX_PIXEL = 640
DISPLAY_LIST_ALL = ['test', 'input', 'DoB', 'mask_black', 'mask_black_dots', 
                    'board', 'board_edge', 'board_gery', 'board_mask_black', 'board_mask_black_dots', 'board_DoB', 'edge_inv', 
                    'lego_rough', 'lego_full', 'lego_dots', 'lego', 'lego_only_color', 'lego_edge', 'lego_correct', 'lego_rect', 'lego_cropped', 'lego_color', 'lego_syn', 'plot_line']
DISPLAY_LIST_TEST = ['board', 'board_n3', 'board_n4', 'lego_full', 'lego', 'lego_cropped', 'lego_color', 'plot_line', 'lego_syn']
DISPLAY_LIST_STREAM = ['board', 'lego_u_edge_S', 'lego_u_edge_norm_L', 'lego_u_dots_L', 'lego_full']
DISPLAY_LIST = DISPLAY_LIST_STREAM if IS_STREAMING else DISPLAY_LIST_TEST
DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500

# Black dots
BD_COUNT_N_ROW = 9
BD_COUNT_N_COL = 16
BD_BLOCK_HEIGHT = IMAGE_HEIGHT / BD_COUNT_N_ROW
BD_BLOCK_WIDTH = IMAGE_WIDTH / BD_COUNT_N_COL
BD_BLOCK_SPAN = max(BD_BLOCK_HEIGHT, BD_BLOCK_WIDTH)
BD_BLOCK_AREA = BD_BLOCK_HEIGHT * BD_BLOCK_WIDTH
BD_COUNT_THRESH = 30
BD_MAX_PERI = (IMAGE_HEIGHT + IMAGE_HEIGHT) / 40
BD_MAX_SPAN = int(BD_MAX_PERI / 4.0 + 0.5)
CHECK_BD_SIZE = 'simple'

# Color detection
# H: hue, S: saturation, B: brightness
# L: lower_bound, U: upper_bound, TH: threshold
HUE_RANGE = 18
BLUE = {'H' : 110, 'S_L' : 100, 'B_TH' : 110} # H: 108
YELLOW = {'H' : 30, 'S_L' : 100, 'B_TH' : 170} # H: 25 B_TH: 180
GREEN = {'H' : 70, 'S_L' : 100, 'B_TH' : 60} # H: 80 B_TH: 75
RED = {'H' : 0, 'S_L' : 100, 'B_TH' : 130}
BLACK = {'S_U' : 80, 'B_U' : 80}
#WHITE = {'S_U' : 60, 'B_L' : 101, 'B_TH' : 160} # this includes side white, too
WHITE = {'S_U' : 60, 'B_L' : 160}
BLACK_DOB_MIN_V = 15
COLOR_ORDER = ['nothing', 'white', 'green', 'yellow', 'red', 'blue', 'black', 'unsure']

# Board
BOARD_MIN_AREA = BD_BLOCK_AREA * 7
BOARD_MIN_LINE_LENGTH = BD_BLOCK_SPAN
BOARD_MIN_VOTE = BD_BLOCK_SPAN / 2
BOARD_RECONSTRUCT_HEIGHT = 155 * 1
BOARD_RECONSTRUCT_WIDTH = 270 * 1
BOARD_BD_MAX_PERI = (BOARD_RECONSTRUCT_HEIGHT + BOARD_RECONSTRUCT_WIDTH) / 30
BOARD_BD_MAX_SPAN = int(BOARD_BD_MAX_PERI / 4.0 + 1.5)
BOARD_RECONSTRUCT_AREA = BOARD_RECONSTRUCT_HEIGHT * BOARD_RECONSTRUCT_WIDTH 
BOARD_RECONSTRUCT_PERI = (BOARD_RECONSTRUCT_HEIGHT + BOARD_RECONSTRUCT_WIDTH) * 2 
BOARD_RECONSTRUCT_CENTER = (BOARD_RECONSTRUCT_HEIGHT / 2, BOARD_RECONSTRUCT_WIDTH / 2)

#Bricks
BRICK_HEIGHT = BOARD_RECONSTRUCT_HEIGHT / 12.25 # magic number
BRICK_WIDTH = BOARD_RECONSTRUCT_WIDTH / 26.2 # magic number
BRICK_HEIGHT_THICKNESS_RATIO = 15 / 12.25
BLOCK_DETECTION_OFFSET = 2

# Optimizations

# If True, performs a second step fine-grained board detection algorithm.
# Depending on the other algorithms, this is usually not needed.
OPT_FINE_BOARD = False

OPT_NOTHING = False
OPT_WINDOW = False
WORST_RATIO_BLOCK_THRESH = 0.55

# If True, do perspective correction first, then color normalization
# If False, do perspective correction after color has been normalized
PERS_NORM = True

def setup(is_streaming):
    global IS_STREAMING, DISPLAY_LIST, DISPLAY_WAIT_TIME, SAVE_IMAGE
    IS_STREAMING = is_streaming
    DISPLAY_LIST = DISPLAY_LIST_STREAM if IS_STREAMING else DISPLAY_LIST_TEST
    DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500
    SAVE_IMAGE = not IS_STREAMING

