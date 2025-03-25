import numpy as np

import pandas as pd

import pickle

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.models import infer_signature


def eval_metrics(actual, prediction):
    rmse = np.sqrt(mean_squared_error(actual, prediction))
    mae = mean_absolute_error(actual, prediction)
    r2 = r2_score(actual, prediction)
    return rmse, mae, r2


def train(cfg):
    train = pd.read_csv(cfg['split_data']['train_path'])
    test = pd.read_csv(cfg['split_data']['test_path'])
    X_train, y_train = (train.drop(columns=[cfg['base']['target']]).values,
                        train[cfg['base']['target']].values)
    X_test, y_test = (test.drop(columns=[cfg['base']['target']]).values,
                      test[cfg['base']['target']].values)

    lasso_pipeline_parameters = {
        'LASSO__max_iter': [1000, 2000, 5000, 7000, 10000, 100000],
        'LASSO__alpha': [1e-6, 1e-5, 1e-4, 1e-2, 1],
        'LASSO__random_state': [42]
    }

    mlflow.set_experiment("linear model cars")
    with mlflow.start_run():
        lasso_pipeline = Pipeline(steps=[
            ('ST_SC', StandardScaler()),
            ('LASSO', Lasso())
        ])

        clf = GridSearchCV(lasso_pipeline,
                           lasso_pipeline_parameters,
                           cv=cfg['train']['cv'],
                           n_jobs=cfg['train']['n_jobs'])

        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_['LASSO']
        y_prediction = best.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, y_prediction)

        final_parameters = best.get_params()
        for parameter in final_parameters:
            mlflow.log_param(parameter, final_parameters[parameter])

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

        with open(cfg['train']['model_path'], "wb") as file:
            pickle.dump(best, file)
