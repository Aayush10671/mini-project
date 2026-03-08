import numpy as np
import pandas as pd
import os
import logging
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug('Model loaded successfully')
        return model
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise


def evaluate(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        logger.debug('Evaluation completed')

        return r2, mae, rmse, predictions

    except Exception as e:
        logger.error('Error during evaluation: %s', e)
        raise


def save_metrics(r2, mae, rmse, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(f"R2 Score: {r2}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"RMSE: {rmse}\n")

        logger.debug('Metrics saved successfully')

    except Exception as e:
        logger.error('Error saving metrics: %s', e)
        raise


def main():
    try:
        logger.debug("Starting model evaluation...")

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("student-performance-evaluation")

        test_data = pd.read_csv('./data/processed/test_engineered.csv')

        X_test = test_data.drop('Final_Score', axis=1)
        y_test = test_data['Final_Score']

        with mlflow.start_run(description="model evaluation"):

            model = load_model('models/model.pkl')

            r2, mae, rmse, predictions = evaluate(model, X_test, y_test)

            print(f"R2 Score: {r2}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")

            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)

            test_df = X_test.copy()
            test_df['Final_Score'] = y_test
            test_df = mlflow.data.from_pandas(test_df)
            mlflow.log_input(test_df, "testing")

            signature = mlflow.models.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(model, "evaluated_model", signature=signature)

            mlflow.set_tag("Author", "aayush")

            save_metrics(r2, mae, rmse, output_path='reports/metrics.txt')

    except Exception as e:
        logger.error('Failed to complete model evaluation: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()