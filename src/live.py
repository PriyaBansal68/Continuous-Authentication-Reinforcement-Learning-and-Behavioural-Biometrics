
import string
import sys
import time
from typing import Dict
import keyboard
import numpy as np
import pandas as pd
from queue import Queue

def get_key_mapping() -> Dict[str, str]:
    mapping = {k:k for k in string.ascii_letters}
    mapping["space"] = " "
    mapping["return"] = "\n"
    mapping["backspace"] = "\b"

    return mapping

def live_compare(df, agent, env):
    """
    Run the live typing and do continuous authentication

    Params
    -------------------------
    by_user: User ID whose data needs to be simulated
    agent: Agent which is used to do contrinuous authentication
    env: Environment to easily get the features for the agent
    """
    hacker_typing = False
    recorded = Queue()
    def switch_mode(*args, **kwargs):
        """Switch mode from user typing to live simulation and vice-versa"""
        nonlocal hacker_typing, recorded
        hacker_typing = not hacker_typing
        env.set_initial_priors()
        if hacker_typing:
            # TODO: before clearing add
            recorded = Queue()
            keyboard.start_recording(recorded)
        else:
            keyboard.stop_recording()
    
    keyboard.on_release_key("left ctrl", switch_mode)

    
    def continuous_authenticate(recordings):
        """
        Continuous authenticate by predicting on a visible observation

        Params
        -------------------------
        recordings: Current unseen recorded data that can be used for prediction
        """
        if len(recordings)<env.No:
            return None
        
        # get first No elements
        observation = pd.DataFrame(recordings[-env.No:], columns=["key","direction","time_diff"])
        observation = env.get_features(observation)

        # remove first Nh elements
        del recordings[:env.Nh]

        # predict
        prediction = agent.predict(observation, exploration=False).item()
        return prediction
    
    # Initialize variables
    all_recordings = []
    key_mapping = get_key_mapping()

    for row in df.itertuples():
        # only updated if hacker_typing
        num_user_updates = 0
        prev_user_time = 0
        while hacker_typing:
            curr_recordings = list(recorded.queue)
            if num_user_updates==0:
                prev_user_time = curr_recordings[0].time
                # remove first element in first update only
                curr_recordings = curr_recordings[1:]
            
            # append all recordings
            for keyevent in curr_recordings:
                all_recordings.append([
                    keyevent.name,
                    1 if keyevent.event_type=="up" else 0,
                    keyevent.time - prev_user_time
                ])
                prev_user_time = keyevent.time
            
            # clear queue
            with recorded.mutex:
                recorded.queue.clear()
            
            # continuous authenticate
            pred = continuous_authenticate(all_recordings)
            if pred and pred==1:
                print("<<HACKER>>")
            
            # Set input frequency to 0.5 s
            # time.sleep(0.5)
            num_user_updates += 1
        
        s = key_mapping.get(row.key, None)
        if s is None:
            s = row.key

        if row.direction==0:
            sys.stdout.write(s)
            sys.stdout.flush()
            time.sleep(row.time_diff)
        
        # record it
        all_recordings.append([
            keyboard.normalize_name(row.key),
            row.direction,
            row.time_diff
        ])

        # continuous authenticate
        pred = continuous_authenticate(all_recordings)
        if pred and pred==1:
            print("<<HACKER>>")
