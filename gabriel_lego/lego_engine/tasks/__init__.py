from typing import Dict, List

import numpy as np

from . import task_Turtle as __task_turtle

task_collection: Dict[str, List[np.ndarray]] = {
    'turtle_head': __task_turtle.bitmaps
}
