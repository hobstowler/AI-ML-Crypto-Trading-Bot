import sys
import os

import pandas as pd
from google.cloud import storage

price_hist_csv = 'rl_price_hist_1h'
bucket_name = 'ai-ml-bitcoin-bot.appspot.com'
kill_thresh = 0.6


def initialize_actor():
    pass


def check_for_models(actor_file: str, critic_file: str):
    actor_file_path = os.path.join(os.getcwd(), actor_file)
    critic_file_path = os.path.join(os.getcwd(), critic_file)
    if os.path.exists(actor_file_path) and os.path.exists(critic_file_path):
        return

    if not os.path.exists(actor_file_path):
        pass

    if not os.path.exists(critic_file_path):
        pass


def check_for_csv() -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), f'{price_hist_csv}.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(price_hist_csv)
        with blob.open("r") as f:
            df = pd.read_csv(f)

        df.to_csv(file_path, index=False)
        return df


def run():
    pass


if __name__ == "__main__":
    print(sys.argv)
    args = sys.argv[1:]
    actor_file_name = args[1]
    critic_file_name = args[2]


