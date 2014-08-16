#!/usr/bin/env python
import cv2
import sys
import time
import argparse
import numpy as np
import bitmap as bm
import lego_cv as lc

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("bitmap",
            help = "The bitmap to convert",
            )
    args = parser.parse_args()
    return args.bitmap

def string2list(string):
    l_ret = []
    if string[0] != '[':
        return int(string)
    s = string[1:-1] # qia tou qu wei
    s += ','
    i = 0
    i_prev = 0
    level = 0
    while i < len(s):
        if s[i] == '[':
            level += 1
            i += 1
        elif s[i] == ']':
            level -= 1
            i += 1
        elif s[i] == ',' and level == 0:
            l = string2list(s[i_prev : i])
            l_ret.append(l)
            i += 1
            i_prev = i
        else:
            i += 1
    return l_ret

if __name__ == "__main__":
    bm_string = parse_arguments()
    bitmap = string2list(bm_string)
    bitmap = np.array(bitmap)
    img_syn = bm.bitmap2syn_img(bitmap)
    cv2.namedWindow('show')
    lc.display_image('show', img_syn, wait_time = 500, is_resize = True, resize_max = -1, resize_scale = 50, save_image = True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt as e:
        sys.stdout.write("user exits\n")
