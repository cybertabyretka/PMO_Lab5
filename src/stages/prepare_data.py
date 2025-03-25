import sys
import os
sys.path.append(os.getcwd())


from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import re
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


def extract_engine_features(engine_str):
    power_match = re.search(r'(\d+\.?\d*)\s*HP', engine_str)
    volume_match = re.search(r'(\d+\.?\d*)\s*L', engine_str)

    power = float(power_match.group(1)) if power_match else None

    volume = float(volume_match.group(1)) if volume_match else None

    remaining_string = engine_str
    if power_match:
        remaining_string = remaining_string.replace(power_match.group(0), '').strip()
    if volume_match:
        remaining_string = remaining_string.replace(volume_match.group(0), '').strip()

    return power, volume, remaining_string


def featurize(df):
    df[['power', 'volume', 'engine']] = df['engine'].apply(
        lambda x: pd.Series(extract_engine_features(x))
    )

    df = df.drop(columns=['milage', 'engine'], axis=1)

    return df


def transform_data(data_path, save=False, save_path=''):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['id'], axis=1)

    df = featurize(df)

    categorical_columns, numerical_columns = get_column_types(df)

    for col in numerical_columns:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)

    for col in categorical_columns:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_value)

    ordinal = OrdinalEncoder()
    ordinal.fit(df[categorical_columns])
    ordinal_encoded = ordinal.transform(df[categorical_columns])
    df_categorical = pd.DataFrame(ordinal_encoded, columns=categorical_columns)

    df[categorical_columns] = df_categorical
    df = df.drop(columns=['clean_title'], axis=1)

    if save:
        df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    config = load_config("./src/config.yaml")
    transform_data(
        config['prepare_data']['path'],
        save=True,
        save_path=config['prepare_data']['save_path']
    )
