"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @created on: 21-09-2022 16:41:47
"""

import argparse
import logging
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from rl_code.models.feature_encoder import KeyboardEncoder
from ..utils.general import initialize_seed
from ..utils.users import get_all_keys, get_all_users
from ..features import preprocess, calc_running_features


class KeyboardFeatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, No: int, p: float = 0.5) -> None:
        super().__init__()

        self.df = df
        self.No = No
        self.n = len(self.df) - self.No + 1
        self.p = p
        self.priors = [0] * len(get_all_keys())

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # get sample
        x = self.df[i : i + self.No]

        # make some augmentations
        y = x.copy()
        if np.random.random() < self.p:
            y["time_diff"] = (
                y["time_diff"]
                * np.random.rand(
                    self.No,
                )
                * 0.03
            )

        # convert into running features
        y, _ = calc_running_features(y, self.priors)
        x, self.priors = calc_running_features(x, self.priors)

        return x, y


def train_encoders(args):
    all_users = (
        [args.user] if args.user is not None else get_all_users(args.dataset_path)
    )
    dataset_path = pathlib.Path(args.dataset_path)

    logging.info(f"Mode: {args.mode}")
    for user_id in all_users:
        # Make path
        user_path = dataset_path / user_id
        keyboard_path = user_path / f"{user_id}_Desktop_Keyboard.csv"

        # Initialize seed
        initialize_seed(args.seed)

        # Read path
        logging.info(f"Reading {keyboard_path}")
        df_key = pd.read_csv(keyboard_path)

        # Preprocess
        df_key = preprocess(df_key)

        # Separate into training and testing
        test_split = 0.25
        split_index = int(len(df_key) * (1 - test_split))

        # Create datasets
        train_dataset = KeyboardFeatDataset(df_key.iloc[:split_index], No=8, p=0.5)
        test_dataset = KeyboardFeatDataset(df_key.iloc[split_index:], No=8, p=0.5)

        # Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128)

        # Initialize model
        model = KeyboardEncoder(in_features=83 * 8, out_features=10, verbose=True)

        # Train model
        model.fit(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=args.epochs,
            lr=args.lr,
        )

        # Save model
        model.save(f"models/{user_id}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="/path/to/dataset/")
    parser.add_argument(
        "--user",
        default=None,
        help="If None, run for all users else run for given user id",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs for training"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Train encoders
    train_encoders(args)
