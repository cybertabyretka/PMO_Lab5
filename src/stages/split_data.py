import sys
import os
sys.path.append(os.getcwd())


from sklearn.model_selection import train_test_split

import pandas as pd

from src.load_cfg import load_config


def data_split(cfg):
    df = pd.read_csv(cfg['split_data']['path'])
    train, test = train_test_split(
        df,
        test_size=cfg['split_data']['test_size']
    )

    train.to_csv(cfg['split_data']['train_path'], index=False)
    test.to_csv(cfg['split_data']['test_path'], index=False)


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    data_split(config)