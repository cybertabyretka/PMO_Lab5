from sklearn.preprocessing import OrdinalEncoder

import pandas as pd

import yaml

from src.load_cfg import load_config


def get_column_types(df):
    categorical_columns = []
    numeric_columns = []

    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)

    return categorical_columns, numeric_columns


def transform_data(data_path, save=False, save_path=''):
    df = pd.read_csv(data_path)
    categorical_columns, _ = get_column_types(df)

    ordinal = OrdinalEncoder()
    ordinal.fit(df[categorical_columns])
    ordinal_encoded = ordinal.transform(df[categorical_columns])
    df_categorical = pd.DataFrame(ordinal_encoded, columns=categorical_columns)

    df[categorical_columns] = df_categorical

    if save:
        df.to_csv(save_path)

    return df


if __name__ == "__main__":
    config = load_config("../config.yaml")
    transform_data(config['data_load']['path'], save=True, save_path=config['data_load']['save_path'])
