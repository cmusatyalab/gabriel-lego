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

IS_STREAMING = False

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# Display
DISPLAY_LIST_ALL = ['input', 'black_dots', 'board', 'board_edge', 'edge_inv', 'board_corrected', 'lego', 'lego_perspective', 'lego_edge', 'lego_correct', 'lego_cropped', 'lego_color', 'lego_syn', 'plot_line']
DISPLAY_LIST_TEST = ['input', 'board_corrected', 'lego_cropped', 'lego_color', 'lego_syn', 'plot_line']
DISPLAY_LIST_STREAM = ['input', 'plot_line', 'lego_syn']
DISPLAY_LIST = DISPLAY_LIST_STREAM if IS_STREAMING else DISPLAY_LIST_TEST
DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500

# Black dots
BD_COUNT_N_ROW = 9
BD_COUNT_N_COL = 16
BD_BLOCK_HEIGHT = IMAGE_HEIGHT / BD_COUNT_N_ROW
BD_BLOCK_WIDTH = IMAGE_WIDTH / BD_COUNT_N_COL
BD_BLOCK_SPAN = max(BD_BLOCK_HEIGHT, BD_BLOCK_WIDTH)
BD_BLOCK_AREA = BD_BLOCK_HEIGHT * BD_BLOCK_WIDTH
BD_COUNT_THRESH = 50
BD_MAX_PERI = (IMAGE_HEIGHT + IMAGE_HEIGHT) / 100
BD_MAX_SPAN = int(BD_MAX_PERI / 4.0 + 0.5)

# Color detection
# H: hue, S: saturation, B: brightness
# L: lower_bound, U: upper_bound, TH: threshold
HUE_RANGE = 7
BLUE = {'H' : 108, 'S_L' : 100, 'B_TH' : 110}
YELLOW = {'H' : 25, 'S_L' : 100, 'B_TH' : 180}
GREEN = {'H' : 80, 'S_L' : 100, 'B_TH' : 75}
RED = {'H' : 4, 'S_L' : 100, 'B_TH' : 130}
BLACK = {'S_U' : 80, 'B_U' : 80}
BLACK_BOARD = {'S_U' : 80, 'B_U' : 110}
#WHITE = {'S_U' : 60, 'B_L' : 101, 'B_TH' : 160} # this includes side white, too
WHITE = {'S_U' : 60, 'B_L' : 160}
WHITE_BOARD = {'S_U' : 60, 'B_L' : 160}
COLOR_ORDER = ['nothing', 'white', 'green', 'yellow', 'red', 'blue', 'black']

# Board
BOARD_MIN_AREA = BD_BLOCK_AREA * 7
BOARD_MAX_DIST2P = BD_BLOCK_SPAN * 2
BOARD_MIN_LINE_LENGTH = BD_BLOCK_SPAN
BOARD_MIN_VOTE = BD_BLOCK_SPAN / 2
BOARD_RECONSTRUCT_HEIGHT = 155
BOARD_RECONSTRUCT_WIDTH = 270

BRICK_HEIGHT = BOARD_RECONSTRUCT_HEIGHT / 12.25 # magic number
BRICK_WIDTH = BOARD_RECONSTRUCT_WIDTH / 26.2 # magic number

DISPLAY_MAX_PIXEL = 640

