

import argparse
from pathlib import Path
from src.data import get_all_valid_users, load_bbmas_data

def preprocess_data(dataset_path):
    all_user_ids = get_all_valid_users(dataset_path)
    data = load_bbmas_data(dataset_path=dataset_path, user_ids=all_user_ids)
    for user_id in data:
        data[user_id]["df"].to_csv(Path.joinpath(Path(dataset_path), str(user_id), f"{user_id}_processed.csv"), index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="/path/to/dataset/")
    args = parser.parse_args()

    preprocess_data(dataset_path=args.dataset_path)
