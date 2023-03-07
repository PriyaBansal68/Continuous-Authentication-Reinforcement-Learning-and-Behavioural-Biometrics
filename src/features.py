

import keyboard
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple
from .utils.keys import get_all_keys, get_key_category


############ Manual feature calculation ################


def calc_running_features(
    data: pd.DataFrame, priors: List[float] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Calculate running features
    [Time diff] + [holdtime per key] for n rows

    Params
    -------------------------
    data: Preprocessed sub dataframe (n rows)
    priors: Previous holdtime per key. If None, reset to 0.
    """
    all_keys = get_all_keys()
    position = {}
    i = 0
    for key in all_keys:
        if get_key_category(key) in position:
            continue
        position[get_key_category(key)] = i
        i += 1
    
    new_feature_data = []
    for i, row in enumerate(data.itertuples()):
        key_cateogry = get_key_category(row.key)
        new_feature_data.append([
            position[key_cateogry],
            row.press_to_press,
            row.release_to_press,
            row.hold_time,
        ])
    

    feature_cols = ["time_diff"]
    running_features = np.concatenate(
        (data[feature_cols].values.reshape(-1, len(feature_cols)), np.array(new_feature_data)), axis=1
    )
    return running_features, priors


def calc_summary_features(data: pd.DataFrame) -> np.ndarray:
    """
    Get summary features (get single aggregated vector for n rows)

    Params
    -------------------------
    data: Preprocessed sub dataframe (n rows)
    """
    # Time length of the events
    col_index = data.columns.get_loc("time_since_beginning")
    time_length = data.iloc[len(data)-1, col_index] - data.iloc[0, col_index]

    # Mouse velocity
    # shifted_data = data.shift(1)
    # avg_mouse_velocity = np.mean(np.sqrt((data["pX"] - shifted_data["pX"])**2 + (data["pY"] - shifted_data["pY"])**2) / data["time_diff"])

    # Space In/Out Hold
    # press_indices = data.loc[data["key"].str.lower()=="space"].index

    # press_indices = [data.iloc[i, data.columns.get_loc("time_diff")] for i in press_indices if i<len(data)]
    # avg_space_in_hold = np.mean(press_indices) if len(press_indices)>0 else 0

    # Space Enter Ratio
    space_ratio = len(data[data["key"].str.lower()=="space"])
    enter_ratio = len(data[data["key"].str.lower()=="return"])

    return np.array([time_length, space_ratio, enter_ratio])


def get_nrf_per_event() -> int:
    """Get number of running features per event"""
    return len(get_all_keys()) + 1


def get_nsf_per_event() -> int:
    """Get number of summary features per event"""
    return 3
