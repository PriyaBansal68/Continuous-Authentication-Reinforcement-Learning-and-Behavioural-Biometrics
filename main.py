"""
 @author    : Priya Bansal
 @contact   : https://www.linkedin.com/in/priyabansal-in/
 @created on: 20-09-2022 
"""

import json
import argparse
import pathlib
import random
import pandas as pd
import torch
from src.data import load_preprocessed_data, get_all_valid_users
from src.agent import Agent
from src.env import BehaviouralBioEnv
from src.runner import Runner
from src.live import live_compare

import logging
logger = logging.getLogger(__name__)

def main(args, config):
    """
    Run the main function

    Params:
        args: Parsed user arguments which contains
            - dataset_path: Path to the dataset so that dataset/{user}/*.csv
            - user: User ID. If None, then runs given mode on all users.
            - mode: Run mode, either of ["train","test","live"]
            - seed: Optional integer number
        config: Dictionary (parsed json configuration file)
    """
    assert (
        args.user is not None or args.mode != "live"
    ), "Cannot run all users for live mode"

    # Get valid user IDs
    all_user_ids = get_all_valid_users(args.dataset_path)

    # Load full data at once
    logger.info("Loading full data...")
    data = load_preprocessed_data(args.dataset_path, all_user_ids)
    logger.info("Loading complete !!")

    user_ids = all_user_ids.copy()
    if args.user is not None:
        user_ids = [args.user]

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Number of users: {len(user_ids)}")
    for user_id in sorted(user_ids):
        logger.info(f"Processing user ID: {user_id}")

        # Find number of available corrupted user IDs for current user
        num_corrupted_users = config["num_corrupted_users"]
        corrupted_user_ids = list(set(all_user_ids)-set([user_id]))
        if num_corrupted_users > len(corrupted_user_ids):
            logging.warning(f"Modifying corrupted users from {num_corrupted_users} to {len(corrupted_user_ids)}")
            num_corrupted_users = len(corrupted_user_ids)
        
        # Put all corrupted data users into a list
        corrupted_user_ids = random.sample(corrupted_user_ids, k=num_corrupted_users)
        corrupted_users_data = [data[corr_user_id]["df"] for corr_user_id in corrupted_user_ids]

        logger.info(f"Selected corrupted user IDs: {corrupted_user_ids}")

        # Split dataset if training or testing
        df = data[user_id]["df"]
        if args.mode=="train":
            # df_keyboard_events = df_keyboard_events[:int(config["train_split"]*len(df_keyboard_events))]
            df_keyboard_events = data[user_id]["df"]
        elif args.mode=="test":
            # df_keyboard_events = df_keyboard_events[int(config["train_split"]*len(df_keyboard_events)):]
            df_keyboard_events = data[user_id]["df"]
        
        # Create environmemt
        env = BehaviouralBioEnv(
            keyboard_events=df_keyboard_events,
            No=config["No"],
            Nh=config["Nh"],
            num_encoder_features=config["num_encoder_features"],
            corrupt_bad_probability=config["corrupt_bad_probability"],
            multi_corrupted_events=corrupted_users_data
        )

        # Agent
        results_path = pathlib.Path(f"output/{args.exp_name}/{user_id}/")
        checkpoint_path = pathlib.Path(results_path / "checkpoint.pt")
        best_path = pathlib.Path(results_path / "best.pt")
        if args.mode=="train":
            model_path = checkpoint_path if args.resume else None
        else:
            model_path = best_path

        agent = Agent(
            model_path=model_path,
            state_features=env.get_num_state_features(),
            num_actions=env.action_space.n,
            eps_start=config["eps_start"],
            eps_end=config["eps_end"],
            eps_decay=config["eps_decay"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Runner
        if args.mode != "live":
            runner = Runner(args.seed)
            runner.run(
                env,
                agent,
                training=True if args.mode=="train" else False,
                num_episodes=config["num_episodes"] if args.mode=="train" else 1,
                c_update=config["c_update"],
                results_savepath=results_path,
            )
        else:
            live_compare(
                df=data[user_id]["df"],
                agent=agent,
                env=env
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="/path/to/dataset/")
    parser.add_argument(
        "--user",
        default=None,
        help="If None, run for all users else run for given user id",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "live"],
        type=str,
        default="test",
        help="Running env mode",
    )
    parser.add_argument("--exp-name", required=True, help="Name of the experiment under which all results will be saved")
    parser.add_argument("--resume", action="store_true", help="resume training from last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Config
    config_path = "config.json"
    config = None
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create output folder
    output_dir = pathlib.Path(f"output/{args.exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run main
    main(args, config)
