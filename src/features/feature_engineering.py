import numpy as np
import pandas as pd
import os
import logging
import mlflow

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def engineer_features(df):
    try:
        # df['Academic_Performance_Index'] = (
        #     df['Midterm_Score'] +
        #     df['Assignments_Avg'] +
        #     df['Quizzes_Avg'] +
        #     df['Projects_Score']
        # ) / 4

        # df['Engagement_Score'] = (
        #     df['Study_Hours_per_Week'] *
        #     df['Attendance (%)']
        # ) / 10

        # df['Wellbeing_Index'] = (
        #     df['Sleep_Hours_per_Night'] -
        #     df['Stress_Level']
        # )
        # df['Participation_Attendance_Ratio'] = (
        #     df['Participation_Score'] /
        #     (df['Attendance (%)'] + 1)
        # )

        logger.debug('Feature engineering completed')
        return df

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        processed_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)

        train_data.to_csv(os.path.join(processed_path, "train_engineered.csv"), index=False)
        test_data.to_csv(os.path.join(processed_path, "test_engineered.csv"), index=False)

        logger.debug(f"Feature engineered data saved to {processed_path}")

    except Exception as e:
        logger.error(f"Error occurred while saving feature engineered data: {e}")
        raise


def main():
    try:
        logger.debug("Starting feature engineering...")

        mlflow.set_experiment("feature_engineering_pipeline")

        with mlflow.start_run():

            train_data = pd.read_csv('./data/interim/train_processed.csv')
            test_data = pd.read_csv('./data/interim/test_processed.csv')
            logger.debug('Data loaded successfully')

            mlflow.log_metric("train_rows_before_fe", train_data.shape[0])
            mlflow.log_metric("test_rows_before_fe", test_data.shape[0])

            train_engineered = engineer_features(train_data)
            test_engineered = engineer_features(test_data)

            mlflow.log_metric("train_rows_after_fe", train_engineered.shape[0])
            mlflow.log_metric("test_rows_after_fe", test_engineered.shape[0])
            mlflow.log_metric("train_columns_after_fe", train_engineered.shape[1])
            mlflow.log_metric("test_columns_after_fe", test_engineered.shape[1])

            save_data(train_engineered, test_engineered, data_path='./data')

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()