

import gym
import random
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Optional


class QPredictor(nn.Module):
    """Prediction network for DDQN"""
    def __init__(self, in_features: int, out_features: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, out_features),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """Replay memory for DDQN"""
    def __init__(self, capacity) -> None:
        self.memory = deque([],maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(
        self,
        model_path: str = None,
        state_features: int = 10,
        num_actions: int = 2,
        eps_start: float = 0.9,
        eps_end: float = 0.01,
        eps_decay: float = 200,
        replay_capacity: int = 10000,
        device = None
    ) -> None:
        """
        DDQN Algorithm (from scratch)

        Params
        -------------------------
        model_path: filepath if provided loads the models.
            If None, initialize agent with random weights and given model parameters.
        """
        self.in_features = state_features
        self.num_actions = num_actions
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.memory = ReplayMemory(replay_capacity)
        self.device = device

        if model_path is None or not pathlib.Path.exists(model_path):
            # Initialize model with random weights
            self.policy_net = QPredictor(
                in_features=self.in_features, out_features=self.num_actions
            )
            self.target_net = QPredictor(
                in_features=self.in_features, out_features=self.num_actions
            )
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        else:
            # Load checkpoint
            checkpoint = torch.load(model_path)
            # self.in_features = checkpoint["in_features"]
            # self.num_actions = checkpoint["num_actions"]

            # Load model weights
            self.policy_net = QPredictor(
                in_features=self.in_features, out_features=self.num_actions
            )
            self.target_net = QPredictor(
                in_features=self.in_features, out_features=self.num_actions
            )
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.01)
        
        # Copy weigths to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.policy_net.to(self.device)
        self.target_net.to(self.device)


    def predict(self, observation, exploration:bool=False):
        """
        Predict the action (0 or 1)

        Params
        -------------------------
        observation: Tensor representing the state
        exploration: Whether to explore using epsilon-greedy algorithm
        """
        # TODO: give an option to return the normalized scores
        # calculate epsilon
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.steps_done / self.eps_decay)

        # choose action
        if exploration and np.random.random() < epsilon:
            action = torch.randint(0, self.num_actions, (1,1)).to(self.device)
            if action[0][0]==0:
                confidence = 0
            else:
                confidence = 1
            self.steps_done += 1
        else:
            with torch.no_grad():
                q_values = self.policy_net(observation.to(self.device))
                action = q_values.max(1)[1].view(1,1)
                confidence = q_values[0][1].cpu().numpy()

        return action, confidence
    
    def record(self, state, action, next_state, reward):
        """Save to the replay memory"""
        self.memory.push(state, action, next_state, reward)

    def optimize(self, batch_size:int=128, discount_factor:float = 0.2):
        """Perform optimizer step on the policy net"""
        if len(self.memory) < batch_size:
            return
        
        # Get replay batch
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Predict q value for given actions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute expected max q values for non-final next states otherwise 0
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
    
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values.view(-1,1) * discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # TODO
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-10, 10)
        self.optimizer.step()

    def update(self):
        """Update the target net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath:str):
        """
        Saves the model

        Params
        -------------------------
        filepath: Saves the model state to the given filepath
        """
        # TODO: save other features like epochs, input size, etc
        output_dir = pathlib.Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.policy_net.state_dict(),
        }, filepath)

