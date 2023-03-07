
import gym
import gym.spaces
import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from .utils.keys import get_all_keys
from .generate import badly_corrupt
from .features import (
    calc_running_features,
    calc_summary_features,
    get_nrf_per_event,
    get_nsf_per_event,
)
from .models.feature_encoder import KeyboardEncoder, KeyboardPCAEncoder

class BehaviouralBioEnv(gym.Env):
    def __init__(
        self,
        keyboard_events: pd.DataFrame = None,
        No: int = 8,
        Nh: int = 3,
        num_encoder_features: int = 10,
        corrupt_bad_probability: float = 0.5,
        multi_corrupted_events: List[pd.DataFrame] = []
    ) -> None:
        """
        Behavioural biometrics gym environment to identify pattern anomalies

        Params
        -------------------------
        keyboard_events: Dataframe describing raw keyboard events
        No: Number of events considered in the observation
        Nh: Number of events hopped (skipped) to take the next step
        num_encoder_features: Number of encoder features
        corrupt_bad_probability: Probability that next observation is corrupted
        """
        super().__init__()

        # Offline data
        self.keyboard_events = keyboard_events
        self.total_len = len(self.keyboard_events)
        self.multi_corrupted_events = multi_corrupted_events

        # Assertions
        assert 0 < Nh <= No <= self.total_len
        assert 0 <= corrupt_bad_probability <= 1

        # Hyperparams
        self.No = No
        self.Nh = Nh
        self.corrupt_bad_probability = corrupt_bad_probability

        # Encoder
        self.num_encoder_features = num_encoder_features
        self.num_summary_features = get_nsf_per_event()

        # FUTURE: With autoencoder
        # self.num_running_features = get_nrf_per_event()
        # self.keyboard_encoder = KeyboardEncoder(
        #     model_path=None,
        #     in_features=self.num_running_features * self.No,
        #     out_features=self.num_encoder_features,
        #     verbose=False,
        # )
        
        self.keyboard_encoder = KeyboardPCAEncoder(df=self.keyboard_events, No=self.No, out_features=self.num_encoder_features)

        # Initial state
        self.start_idx = 0
        self.last_corrupted = False
        self.set_initial_priors()

        # Space
        self.action_space = gym.spaces.Discrete(2)

    def get_num_state_features(self) -> int:
        """Get number of total state features to be output"""
        return self.num_encoder_features + self.num_summary_features
    
    def set_initial_priors(self):
        """Set initial priors (holdtime) to be 0"""
        self.priors = [0] * len(get_all_keys())
    
    def get_raw_data(self, start) -> Tuple[np.array, bool]:
        """
        Get raw data which may be corrupted indicated by flag

        Params
        -------------------------
        start: Gives the start index of keyboard events
        """
        corrupted = False
        data = self.keyboard_events[start : start + self.No]
        if np.random.random() < self.corrupt_bad_probability:
            # corrupt data
            corrupted = True
            data = badly_corrupt(data, self.multi_corrupted_events)
            
            # assume that priors reset in corruption
            self.set_initial_priors()
        
        # Also reset if coming from corrupt to positive
        if not corrupted and self.last_corrupted:
            self.set_initial_priors()

        return data, corrupted

    def get_features(self, data):
        """
        Get features from the raw data

        Params
        -------------------------
        data: Preprocessed sub dataframe
        """
        # running features
        running_features, self.priors = calc_running_features(data, self.priors)

        # encode running features
        features = self.keyboard_encoder.predict(running_features)

        # summary_features
        summary_features = calc_summary_features(data)

        # Concatenate both lists
        features = torch.tensor(np.concatenate((features, summary_features), axis=0), dtype=torch.float32).unsqueeze(0)

        return features

    def reset(self) -> np.array:
        """Resets the environment and returns the observation"""
        # Reset state
        self.start_idx = 0
        self.last_corrupted = False
        self.set_initial_priors()

        # Get raw data
        data, self.last_corrupted = self.get_raw_data(self.start_idx)

        # Get observation
        observation = self.get_features(data)

        return observation

    def step(self, action: int) -> Tuple[np.array, float, bool, bool]:
        """
        Give a reward to the performed action and move to the next observation by Nh hops

        Params
        -------------------------
        action: user or hacker (0 or 1) 
        """
        # Get ground truth for last observation
        y_true = 1.0 if self.last_corrupted else 0.0

        # Get reward
        reward, metric_cls = self.reward_system(y_true, action)

        # Update to next state
        self.start_idx += self.Nh
        self.last_corrupted = False

        # Terminal state
        terminated = False  # inside of MDP
        truncated = False  # outside of MDP
        if self.start_idx + self.No > self.total_len:
            truncated = True
        # TODO: Question is should we terminate if we get continuous 1s

        # Get next observation
        observation = None
        if not terminated and not truncated:
            data, self.last_corrupted = self.get_raw_data(self.start_idx)
            observation = self.get_features(data)

        return observation, reward, terminated, truncated, {'metric_cls':metric_cls, 'y_true': y_true}

    def reward_system(self, y_true: float, y_pred: float) -> float:
        """
        Reward system given predicted action and ground truth (i.e. corrupted or not)

        Params
        -------------------------
        y_true: Ground truth (corrupted or not)
        y_pred: Predicted action (user or hacker)
        """
        if y_true == y_pred:
            # TP, TN
            return 1.0, "TP" if y_true==1 else "TN"
        else:
            # FP, FN
            return 0.0, "FP" if y_true==0 else "FN"
