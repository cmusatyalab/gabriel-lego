from typing import Dict, List

import numpy as np

from . import task_Turtle as __task_turtle, task_generated_135 as __task_135, \
    task_generated_180 as __task_180, task_generated_20 as __task_20, \
    task_generated_270 as __task_270, task_generated_45 as __task_45, \
    task_generated_90 as __task_90

task_collection: Dict[str, List[np.ndarray]] = {
    'turtle_head'  : __task_turtle.bitmaps,
    'generated_20' : __task_20.bitmaps,
    'generated_45' : __task_45.bitmaps,
    'generated_90' : __task_90.bitmaps,
    'generated_135': __task_135.bitmaps,
    'generated_180': __task_180.bitmaps,
    'generated_270': __task_270.bitmaps
}
