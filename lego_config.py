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

DISPLAY_LIST_ALL = ['input', 'board', 'board_edge', 'board_corrected', 'lego', 'lego_perspective', 'lego_edge', 'lego_correct', 'lego_cropped', 'lego_syn', 'plot_line']
DISPLAY_LIST_TEST = ['board', 'board_corrected', 'lego_perspective', 'board_edge', 'lego_syn'] #, 'plot_line']
DISPLAY_LIST_STREAM = ['input', 'lego_cropped', 'lego_syn']

if IS_STREAMING:
    DISPLAY_LIST = DISPLAY_LIST_STREAM
else:
    DISPLAY_LIST = DISPLAY_LIST_TEST

BOARD_RECONSTRUCT_HEIGHT = 155
BOARD_RECONSTRUCT_WIDTH = 270

BRICK_HEIGHT = BOARD_RECONSTRUCT_HEIGHT / 12.25 # magic number
BRICK_WIDTH = BOARD_RECONSTRUCT_WIDTH / 26.2 # magic number

DISPLAY_MAX_PIXEL = 640

