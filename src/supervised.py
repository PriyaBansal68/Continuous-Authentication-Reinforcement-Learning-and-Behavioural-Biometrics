
import torch
import numpy as np
from itertools import count
from .utils.general import initialize_seed
from sklearn.metrics import classification_report
from .metrics import calculate_eer


class Trainer:
    def __init__(self, seed: int = 42) -> None:
        """Create one runner for one environment"""
        initialize_seed(seed)

    def run(self, env, model, training:bool=False, num_episodes: int = 50, c_update: int = 4, batch_size: int = 128, savefile=None):
        """
        Runner code to complete the loop

        Params
        -------------------------
        env: Gym environment
        agent: Agent to run in the given environment
        training: Whether to train the agent or just test it
        num_episodes: Number of episodes to run the agent
        c_update: After how many updates should the target net update in the agent [Valid when training is True]
        batch_size: Batch size used for training the agent [Valid when training is True]
        savefile: Save path to save the agent
        """
        X = []
        y = []
        for episode_i in range(1, num_episodes+1):
            observation = env.reset()

            # Run episode
            for t in count():
                X.append(observation.numpy())
                
                # Step environment
                observation, reward, terminated, truncated, info = env.step(1)

                if reward==1:
                    y.append(1)
                else:
                    y.append(0)
                
                # Exit condition
                if terminated or truncated:
                    break
        
        X = np.array(X).squeeze(axis=1)
        y = np.array(y)
        
        model.fit(X[:int(0.7*len(X))], y[:int(0.7*len(X))])
        
        y_pred = model.predict(X[int(0.7*len(X)):])
        print(y_pred)
        print(classification_report(y[int(0.7*len(X)):], y_pred))
        print(calculate_eer(y[int(0.7*len(X)):], y_pred))
