import os
import sys
import numpy as np
import pandas as pd
import pymongo

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

from sklearn.model_selection import train_test_split

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        try:
            if MONGO_DB_URL is None:
                raise Exception("MONGO_DB_URL not found. Check .env file")

            print("Mongo URL:", MONGO_DB_URL)

            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            print("DB:", database_name)
            print("Collection:", collection_name)

            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            print("Fetched rows:", len(df))

            if df.empty:
                raise Exception("No data found in MongoDB collection")

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace(["na", "NA", "NaN", ""], np.nan, inplace=True)

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            dataframe.to_csv(file_path, index=False, header=True)
            logging.info("Feature store created")

            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            if dataframe.empty:
                raise Exception("Dataframe is empty before splitting")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info("Train-test split completed")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)