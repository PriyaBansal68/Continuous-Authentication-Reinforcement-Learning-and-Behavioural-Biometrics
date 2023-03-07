"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @created on: 20-09-2022 07:56:49
"""

import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Optional
from sklearn.decomposition import PCA

from ..features import calc_running_features
from ..utils.keys import get_all_keys

class _KeyboardEncoder(nn.Module):
    # TODO: Implement autoencoder
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, X):
        return X[:, : self.out_features]


class KeyboardEncoder:
    def __init__(
        self,
        model_path: Optional[str] = None,
        in_features: int = 83,
        out_features: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Wraps the keyboard feature encoder model

        Params
        -------------------------
        model_path: Filepath if provided loads the models and nullifies other model parameters.
            If None, initialize feature encoder with random weights and given model parameters.
        in_features: If model_path is None, sets the number of input features.
        out_features: If model_path is None, sets the number of output features.
        verbose: Whether to show log information
        """
        self.in_features = in_features
        self.out_features = out_features
        self.verbose = verbose

        if model_path is None:
            # Initialize model with random weights
            self.model = _KeyboardEncoder(
                in_features=self.in_features, out_features=self.out_features
            )

        else:
            # Load checkpoint
            checkpoint = torch.load(model_path)
            self.in_features = checkpoint["in_features"]
            self.out_features = checkpoint["out_features"]

            # Load model weights
            self.model = _KeyboardEncoder(
                in_features=self.in_features, out_features=self.out_features
            )
            self.model.load_state_dict(checkpoint["state_dict"])

    def fit(
        self, train_dataloader, test_dataloader=None, epochs: int = 100, lr: int = 0.001
    ) -> None:
        """
        Fit the autoencoder model

        Params
        -------------------------
        train_dataloader:
        test_dataloader:
        epochs: Number of epochs
        lr: Learning rate
        """
        # Define criterion
        criterion = None

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Training loop
            self.model.train()
            for inputs, targets in train_dataloader:
                optimizer.zero_grad()

                y_pred = self.model.predict(inputs)
                loss = criterion(targets, y_pred)

                loss.backward()
                optimizer.step()

            # Validation check
            if test_dataloader is not None:
                self.model.eval()
                for inputs, targets in train_dataloader:
                    y_pred = self.model.predict(inputs)
                    loss = criterion(targets, y_pred)

    def predict(self, x):
        x = torch.tensor(x.ravel().reshape(1, -1))
        y = self.model(x)
        y = y.squeeze(axis=0).numpy()

        return y

    def save(self, filepath: str) -> None:
        """Saves the model in given filepath"""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "out_features": self.out_features,
            "in_features": self.in_features,
        }

        torch.save(checkpoint, filepath)
        if self.verbose:
            logging.info(f"Saved model at {filepath}")
    

class KeyboardPCAEncoder:
    def __init__(
        self,
        df: pd.DataFrame,
        No: int = 8,
        out_features: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Wraps the keyboard feature encoder model

        Params
        -------------------------
        df: Input preprocessed data
        No: Number of events in observation
        out_features: If model_path is None, sets the number of output features.
        verbose: Whether to show log information
        """
        self.out_features = out_features
        self.verbose = verbose

        self.model = PCA(n_components=self.out_features)
        self._fit(self.model, df, No)
    
    def _fit(self, model, df, No):
        priors = [0] * len(get_all_keys())
        X = []
        for i in range(len(df) - No + 1):
            x = df.iloc[i:i+No]
            x, priors = calc_running_features(x, priors)
            X.append(x.ravel())
        
        model.fit(np.array(X))
    
    def predict(self, x):
        return self.model.transform(np.expand_dims(np.ravel(x), axis=0))[0]

