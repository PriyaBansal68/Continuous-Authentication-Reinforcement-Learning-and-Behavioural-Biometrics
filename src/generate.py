

import random
import numpy as np
import pandas as pd
from typing import List

def badly_corrupt(data: pd.DataFrame, multi_corrupted_events: List[pd.DataFrame]):
    # Choose corrupted event
    corrupted_events = random.choice(multi_corrupted_events)

    # Select the same length of data
    n = len(data)
    i = random.randint(0, len(corrupted_events)-n-1)
    corrupted = corrupted_events[i:i+n].copy()
    corrupted["time_diff"] += np.random.random() * 0.05 - 0.025
    return corrupted


def augment(data):
    # Small random noise to original
    pass
