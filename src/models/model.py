import numpy as np
import pandas as pd
import os
import logging
import yaml
import pickle
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_training_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise


def train_model(X_train, y_train, param_distributions):
    try:
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=50,
            scoring='r2',
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        logger.debug('Best parameters found: %s', random_search.best_params_)
        return random_search

    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, model_path: str):
    try:
        os.makedirs(model_path, exist_ok=True)

        with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)

        logger.debug('Model saved successfully using pickle')

    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise


def main():
    try:
        logger.debug("Starting model training...")

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("student-performance-xgb")

        params = load_params('params.yaml')
        param_distributions = params['model_training']['param_distributions']

        train_data = pd.read_csv('./data/processed/train_engineered.csv')

        leakage_columns = [
            col for col in train_data.columns
            if col.startswith('Grade_') or col.startswith('Pass/Fail_')
        ]

        train_data = train_data.drop(columns=leakage_columns)

        X_train = train_data.drop('Final_Score', axis=1)
        y_train = train_data['Final_Score']

        with mlflow.start_run(description="xgboost randomized search") as parent:

            random_search = train_model(X_train, y_train, param_distributions)

            for i in range(len(random_search.cv_results_['params'])):
                with mlflow.start_run(nested=True):
                    mlflow.log_params(random_search.cv_results_['params'][i])
                    mlflow.log_metric("r2_score", random_search.cv_results_['mean_test_score'][i])

            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = random_search.best_score_

            mlflow.log_params(best_params)
            mlflow.log_metric("best_r2_score", best_score)

            train_df = X_train.copy()
            train_df['Final_Score'] = y_train
            train_df = mlflow.data.from_pandas(train_df)
            mlflow.log_input(train_df, "training")

            signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))

            mlflow.sklearn.log_model(best_model, "xgboost_model", signature=signature)

            mlflow.set_tag("Author", "aayush")

            save_model(best_model, model_path='models')

    except Exception as e:
        logger.error('Failed to complete the model training process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()