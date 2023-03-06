import sys
import os

import pandas as pd
from google.cloud import storage

from RL_Model.misc.secret_client import SecretClient

price_hist_csv = 'rl_price_hist_1h'
actor_model_name = 'actor_ppo_1h'
critic_model_name = 'critic_ppo_1h'
bucket_name = 'ai-ml-bitcoin-bot.appspot.com'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
kill_thresh = 0.6

secret_client = SecretClient()
binance_secret = secret_client.get_secret("binance-api-secret-hobs")
binance_key = secret_client.get_secret("binance-api-key-hobs")


def initialize_actor():
    pass


def check_for_models():
    actor_file_path = os.path.join(os.getcwd(), 'models', actor_model_name)
    critic_file_path = os.path.join(os.getcwd(), 'models', actor_model_name)
    if os.path.exists(actor_file_path) and os.path.exists(critic_file_path):
        return

    if not os.path.exists(actor_file_path):
        blob = bucket.blob(actor_model_name)
        if blob.exists():
            blob.download_to_filename(actor_file_path)
        else:
            # TODO
            raise FileNotFoundError

    if not os.path.exists(critic_file_path):
        blob = bucket.blob(critic_model_name)
        if blob.exists():
            blob.download_to_filename(critic_file_path)
        else:
            # TODO
            raise FileNotFoundError


def generate_price_hist_csv():
    pass


def check_for_price_hist_csv():
    """
    Checks for a local copy of the price history CSV and either downloads it or generates it from binance data
    :return:
    """
    file_path = os.path.join(os.getcwd(), f'{price_hist_csv}.csv')
    if not os.path.exists(file_path):
        blob = bucket.blob(price_hist_csv)
        if blob.exists():
            with blob.open("r") as f:
                df = pd.read_csv(f)
            df.to_csv(file_path, index=False)
        else:
            generate_price_hist_csv()

    return #df


def check_files():
    check_for_price_hist_csv()
    check_for_models()


def prep_data():
    pass


def get_quote():
    pass


def run():
    check_files()
    prep_data()
    get_quote()


def train():
    pass


if __name__ == "__main__":
    print(sys.argv)
    args = sys.argv[1:]
    if args[1] == 'train':
        train()
    else:
        run()


