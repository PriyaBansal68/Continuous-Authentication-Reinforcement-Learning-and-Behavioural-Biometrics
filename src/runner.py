
import torch
import pickle
from pathlib import Path
import numpy as np
from itertools import count
from typing import Optional
from .utils.general import initialize_seed
from .metrics import calculate_eer, calculate_far_frr
import logging

class Runner:
    def __init__(self, seed: int = 42) -> None:
        """Create one runner for one environment"""
        initialize_seed(seed)

    def run(
        self,
        env,
        agent,
        training:bool=False,
        num_episodes: int = 50,
        c_update: int = 4,
        batch_size: int = 128,
        results_savepath:Optional[str]=None,
    ):
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
        
        best = {
            "eer": 1,
            "far": 1,
            "frr": 1,
            "accuracy": 0,
            "iteration": 0,
            "y_true": [],
            "y_pred": [],
            "cm": {}
        }

        eer_per_iteration = []
        # Outer episode loop
        for episode_i in range(1, num_episodes+1):
            observation = env.reset()
            total_reward = 0
            cm = {"TP":0, "FP":0, "TN":0, "FN":0}
            y_true = []
            y_pred = []
            # Run episode
            for t in count():
                # Predict action
                action, confidence = agent.predict(observation, exploration=training)
                
                # Step environment
                next_observation, reward, terminated, truncated, info = env.step(action)

                reward = torch.tensor(reward, dtype=torch.float32).view(1,1)

                if training:
                    # Push to memory
                    agent.record(observation, action, next_observation, reward)

                    # Sample relay memory and optimize agent
                    agent.optimize(batch_size=batch_size)
                
                # Move to next state
                observation = next_observation

                # Debugging params
                total_reward += reward.item()
                cm[info["metric_cls"]] += 1

                y_true.append(info["y_true"])
                y_pred.append(float(confidence))

                # Exit condition
                if terminated or truncated:
                    # Calculate metrics
                    accuracy = total_reward / (t + 1)
                    eer = calculate_eer(y_true, y_pred)
                    far, frr = calculate_far_frr(cm)
                    
                    eer_per_iteration.append(eer)

                    # Save models
                    if training and results_savepath:
                        agent.save(Path(results_savepath / "checkpoint.pt"))

                    # Update best
                    if eer <= best["eer"] and results_savepath:
                        best["accuracy"] = accuracy
                        best["eer"] = eer
                        best["far"] = far
                        best["frr"] = frr
                        best["iteration"] = episode_i
                        best["y_true"] = y_true
                        best["y_pred"] = y_pred
                        best["cm"] = cm

                        agent.save(Path(results_savepath / "best.pt"))
                    
                    logging.info(f"Episode length: {t+1}; reward: {total_reward}; cm: {cm}; exploration: {agent.steps_done}; eer: {eer:.2f};far: {far:.2f};frr: {frr:.2f}")
                    break
            
            # Update agent
            if training and episode_i % c_update == 0:
                agent.update()

        # Save results
        logging.info(f"Metrics on best EER iteration: EER={best['eer']:.4f}, Accuracy={best['accuracy']:.4f}")

        metrics = {}
        if training:
            metrics_savepath = Path(results_savepath / "train_metrics.pickle")
            metrics["eer_per_iteration"] = eer_per_iteration
        else:
            metrics_savepath = Path(results_savepath / "test_metrics.pickle")
        
        metrics["best_metrics"] = best
        with open(metrics_savepath, "wb") as f:
            pickle.dump(metrics, f)
