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


def featurize(df, cfg):
    df['Distance_by_year'] = df['Distance'] / (2022 - df['Year'])
    df['age'] = 2024 - df['Year']
    mean_engine_cap = df.groupby('Style')['Engine_capacity(cm3)'].mean()
    df['eng_cap_diff'] = df.apply(lambda row: abs(row['Engine_capacity(cm3)'] - mean_engine_cap[row['Style']]), axis=1)

    max_engine_cap = df.groupby('Style')['Engine_capacity(cm3)'].max()
    df['eng_cap_diff_max'] = df.apply(lambda row: abs(row['Engine_capacity(cm3)'] - max_engine_cap[row['Style']]), axis=1)

    df.to_csv(cfg['prepare_data']['save_path'], index=False)


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
    df = transform_data(config['data_load']['path'])
    featurize(df, config)
