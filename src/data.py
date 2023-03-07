
import keyboard
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

import logging
logger = logging.getLogger(__name__)

def get_all_valid_users(dataset_path: str) -> List[str]:
    """Get all user ids given the dataset path"""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"{dataset_path} not found or path broken")

    all_user_ids = []
    for f in dataset_path.iterdir():
        if f.is_dir():
            keyboard_path = Path(f / f"{f.name}_Desktop_Keyboard.csv")
            mouse1_path = Path(f / f"{f.name}_Mouse_Button.csv")
            mouse2_path = Path(f / f"{f.name}_Mouse_Move.csv")
            mouse3_path = Path(f / f"{f.name}_Mouse_Wheel.csv")
            if keyboard_path.exists() and mouse1_path.exists() and mouse2_path.exists() and mouse3_path.exists():
                all_user_ids.append(f.name)
            
            else:
                logger.warn(f"Dropped user ID {f.name} because of missing keyboard or mouse data")

    return all_user_ids

def load_preprocessed_data(dataset_path: str, user_ids: List[str]):
    dataset_path = Path(dataset_path)

    data = {}
    for user_id in user_ids:
        # Read
        data[user_id] = {"df": pd.read_csv(Path(dataset_path / "processed_data" / f"{user_id}_processed.csv"))}
    
    return data

def load_bbmas_data(dataset_path: str, user_ids: List[str]):
    dataset_path = Path(dataset_path)

    data = {}
    for user_id in user_ids:
        # Read
        df_keyboard = pd.read_csv(Path(dataset_path / user_id / f"{user_id}_Desktop_Keyboard.csv"))
        df_mouse1 = pd.read_csv(Path(dataset_path / user_id / f"{user_id}_Mouse_Button.csv"))
        df_mouse2 = pd.read_csv(Path(dataset_path / user_id / f"{user_id}_Mouse_Move.csv"))
        df_mouse3 = pd.read_csv(Path(dataset_path / user_id / f"{user_id}_Mouse_Wheel.csv"))

        # Filter out cols
        df_mouse1 = df_mouse1[["pX", "pY", "LR", "state", "time"]]
        df_mouse2 = df_mouse2[["pX", "pY", "time"]]
        df_mouse3 = df_mouse3[["pX", "pY", "delta", "time"]]
        df_keyboard = df_keyboard[["key", "direction", "time"]]

        # Add operation col (denotes mouse-0 or keyboard-1)
        df_keyboard["operation"] = 1

        # Merge mouse data
        df_mouse = pd.merge(df_mouse1, df_mouse2, how="outer", on="time", suffixes=("","_y"))
        maskx, masky = df_mouse["pX_y"].isna(), df_mouse["pY_y"].isna()
        df_mouse.loc[~maskx, "pX"] = df_mouse.loc[~maskx, "pX_y"]
        df_mouse.loc[~masky, "pY"] = df_mouse.loc[~masky, "pY_y"]
        df_mouse.drop(["pX_y","pY_y"], axis=1, inplace=True)

        df_mouse = pd.merge(df_mouse, df_mouse3, how="outer", on="time", suffixes=("","_y"))
        maskx, masky = df_mouse["pX_y"].isna(), df_mouse["pY_y"].isna()
        df_mouse.loc[~maskx, "pX"] = df_mouse.loc[~maskx, "pX_y"]
        df_mouse.loc[~masky, "pY"] = df_mouse.loc[~masky, "pY_y"]
        df_mouse.drop(["pX_y","pY_y"], axis=1, inplace=True)

        # Merge keyboard data
        df = pd.merge(df_keyboard, df_mouse, how="outer", on="time", sort=True).reset_index(drop=True)

        # Type convert time
        df["time"] = pd.to_datetime(df["time"])
        
        # Fill delta with 0 and then forward fill
        df["delta"] = df["delta"].fillna(0)
        df["operation"] = df["operation"].fillna(0)
        df = df.fillna(method="ffill")

        # Drop non-keyboard data from the start and end
        first_valid_key_index = df["key"].first_valid_index()
        last_valid_key_index = df["key"].last_valid_index()
        df.drop(list(range(first_valid_key_index)) + list(range(last_valid_key_index+1,len(df))), inplace=True)

        # Fill other values
        df.loc[df["pX"].isna(), "pX"] = df["pX"].mean().round(0)
        df.loc[df["pY"].isna(), "pY"] = df["pY"].mean().round(0)
        df["LR"] = df["LR"].fillna(0)
        df["state"] = df["state"].fillna(1)

        # Normalise key name
        df["key"] = df["key"].astype(str).apply(keyboard.normalize_name)

        # Add time difference
        df["time_diff"] = (df["time"] - df["time"].shift(1)).transform(
            lambda x: x.total_seconds()
        )
        df.iloc[0, df.columns.get_loc("time_diff")] = 0
        df["time_since_beginning"] = np.cumsum(df["time_diff"])

        # Prepare key ids
        pressed_keys = {}
        curr_keys = set()
        key_ids = []
        key_id = 0
        for row in df.itertuples():
            key_lower = row.key.lower()
            if row.operation==1:
                if row.direction==0:
                    if key_lower not in curr_keys:
                        curr_keys.add(key_lower)
                        pressed_keys[key_lower] = key_id
                        key_id += 1
                else:
                    curr_keys.discard(key_lower)
            
            key_ids.append(pressed_keys.get(key_lower, -1))
        
        # Remove elements with -1 key id
        df["key_id"] = key_ids
        df = df[df["key_id"]!=-1]
        
        df_press_release = df.groupby("key_id").apply(pr_features)
        df_press_release = df_press_release.dropna().reset_index(level=1, drop=True)

        # Create press-release features
        df_press_release["press_to_press"] = np.nan
        df_press_release["release_to_press"] = np.nan
        df_press_release["hold_time"] = np.nan

        for i in range(1, len(df_press_release)):
            press_col_i = df_press_release.columns.get_loc("press")
            release_col_i = df_press_release.columns.get_loc("release")
            press_to_press_i = df_press_release.columns.get_loc("press_to_press")
            release_to_press_i = df_press_release.columns.get_loc("release_to_press")
            hold_time_i = df_press_release.columns.get_loc("hold_time")

            prev_press = df.loc[df_press_release.iloc[i-1, press_col_i], "time_since_beginning"]
            prev_release = df.loc[df_press_release.iloc[i-1, release_col_i], "time_since_beginning"]
            curr_press = df.loc[df_press_release.iloc[i, press_col_i], "time_since_beginning"]
            curr_release = df.loc[df_press_release.iloc[i, release_col_i], "time_since_beginning"]

            df_press_release.iat[i, press_to_press_i] = curr_release - curr_press
            df_press_release.iat[i, release_to_press_i] = curr_press - prev_release
            df_press_release.iat[i, hold_time_i] = curr_press - prev_press
        
        df_press_release = df_press_release.dropna()

        # Remove intm mouse data
        df = pd.merge(df, df_press_release, left_index=True, right_on="press")

        # Drop columns
        df = df.drop(["time", "direction", "press", "release"], axis=1).sort_values("time_since_beginning")
        
        # Temp sacrifice
        df = df[df["time_diff"]<10]
        df = df[:int(len(df)*0.3)]

        # Store
        data[user_id] = {
            "df": df
        }

    return data

def pr_features(g):
    pressed = g[g["direction"]==0]
    released = g[g["direction"]==1]

    pressed = pressed[pressed["time_since_beginning"].min() == pressed["time_since_beginning"]]
    released = released[released["time_since_beginning"].min() == released["time_since_beginning"]]

    if len(pressed)==0 or len(released)==0:
        return pd.DataFrame({"press":[np.nan], "release":[np.nan]})
    return pd.DataFrame({"press":[pressed.index[0]], "release":[released.index[0]]})
