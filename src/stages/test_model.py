import sys
import os
sys.path.append(os.getcwd())


from src.load_cfg import load_config
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dvclive import Live


def test_model(cfg):
    with open(cfg['test']['model_path'], 'rb') as file:
        model = pickle.load(file)

    # with open(cfg['test']['power_path'], 'rb') as file:
    #     power = pickle.load(file)

    df_test = pd.read_csv(cfg['test']['test_path'])
    X_test, y_test = (df_test.drop(columns=[cfg['base']['target']]).values,
                      df_test[cfg['base']['target']].values)

    # y_pred = power.inverse_transform(model.predict(X_test).reshape(-1, 1))
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    with Live() as live:
        live.log_metric("mae", mae)
    return mae


if __name__ == '__main__':
    config = load_config('./src/config.yaml')
    mae = test_model(config)
    print(f'MAE: {mae}')
