
import argparse
import string
import os
import sys
import time
import keyboard
import numpy as np
import pandas as pd
from queue import Queue
from datetime import datetime, timedelta


def record_keyboard_data(output_filepath:str):
    # Initialize variables
    all_recordings = []
    eid = 0
    input_frequency = 0.5

    recorded = Queue()
    keyboard.start_recording(recorded)
    start_datetime = datetime.now()
    while True:
        curr_recordings = list(recorded.queue)
        
        # append all recordings
        escape_pressed = False
        for keyevent in curr_recordings:
            if keyevent.name=="esc":
                escape_pressed = True
                break
            all_recordings.append([
                eid,
                keyevent.name,
                1 if keyevent.event_type=="up" else 0,
                keyevent.time
            ])
            eid += 1
        
        if escape_pressed:
            break

        # clear queue
        with recorded.mutex:
            recorded.queue.clear()
        
        # Set input frequency
        time.sleep(input_frequency)
    
    keyboard.stop_recording()

    # Save recordings
    first_recorded_time = all_recordings[0][-1]
    all_recordings = pd.DataFrame(all_recordings, columns=["eid","key","direction","time"])
    all_recordings["time"] = all_recordings["time"].apply(lambda x: start_datetime + timedelta(seconds=x - first_recorded_time, ))

    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))
    all_recordings.to_csv(output_filepath, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="/path/to/output.csv")
    args = parser.parse_args()

    # Input prompt
    print("Open text editor to avoid typing in terminal!!\n")
    print("Press escape to stop recording!!\n")
    sys.stdout.write(f"Start typing in ")
    sys.stdout.flush()
    timer = 3
    for i in range(timer,-1,-1):
        s = f"{i}.."
        n = len(s)
        sys.stdout.write(s)
        sys.stdout.flush()

        if i==0:
            print(end="\n\n")
            break

        time.sleep(1)
        sys.stdout.write("\b"*n)
        sys.stdout.flush()
    
    # Record
    record_keyboard_data(args.output)
