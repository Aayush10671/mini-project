import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import mlflow
import pickle

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_dataframe(df):
    try:

        ordinal_cols = [
            'Difficulty_Level',
            'Parent_Education_Level',
            'Family_Income_Level'
        ]

        # for col in ordinal_cols:
        #     if col in df.columns:
        #         le = LabelEncoder()
        #         df[col] = le.fit_transform(df[col])

       

        for col in ordinal_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

                os.makedirs("encoders", exist_ok=True)
                with open(f"models/{col}_label_encoder.pkl", "wb") as f:
                    pickle.dump(le, f)
                

        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
            encoded_df.index = df.index
            df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

        logger.debug('Encoding completed')
        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")

    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def main():
    try:
        logger.debug("Starting data preprocessing...")

        mlflow.set_experiment("data_preprocessing_pipeline")

        with mlflow.start_run():

            train_data = pd.read_csv('./data/raw/train.csv')
            test_data = pd.read_csv('./data/raw/test.csv')
            logger.debug('Data loaded successfully')

            mlflow.log_metric("train_rows_before", train_data.shape[0])
            mlflow.log_metric("test_rows_before", test_data.shape[0])

            train_processed_data = preprocess_dataframe(train_data)
            test_processed_data = preprocess_dataframe(test_data)

            mlflow.log_metric("train_rows_after", train_processed_data.shape[0])
            mlflow.log_metric("test_rows_after", test_processed_data.shape[0])
            mlflow.log_metric("train_columns_after", train_processed_data.shape[1])
            mlflow.log_metric("test_columns_after", test_processed_data.shape[1])

            save_data(train_processed_data, test_processed_data, data_path='./data')

    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()