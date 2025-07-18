
from network_sec.exception.exception import NetworkSecurityException
from network_sec.logs.logger import logging


## configuration of the Data Ingestion Config

from network_sec.entity.config_entity import dataIngestionConfig
from network_sec.entity.artifacts_entity import dataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()


MONGO_DB_URL = os.getenv("MONGODB_CLUSTER_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config : dataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def export_dataframe(self):
        try : 
            data_ingestion = self.data_ingestion_config
            client = pymongo.MongoClient(MONGO_DB_URL)

            print(data_ingestion.database +"   "+ data_ingestion.collection)

            collection = client[data_ingestion.database][data_ingestion.collection]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            
            df.replace({"na":np.nan},inplace=True)

            if df.empty:
                raise ValueError("No data found in the MongoDB collection. The DataFrame is empty.")


            logging.info("fetching the information from the database")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_to_feature_store(self, dataframe : pd.DataFrame):

        try:    
            ingested_dir = self.data_ingestion_config.ingested
            dir_path = os.path.dirname(ingested_dir)
            os.makedirs(dir_path, exist_ok=True)
            
            dataframe.to_csv(ingested_dir, index=False, header=True)

            logging.info("Exported to Feature store directory")

            return dataframe
        except Exception as e : 
            raise NetworkSecurityException(e , sys)
        
    def export_to_ingested_dir(self, dataframe : pd.DataFrame):

        try:
            logging.info("initiating Data Ingestion")
            train_dir = self.data_ingestion_config.train_store
            test_dir = self.data_ingestion_config.test_store

            train_path = os.path.dirname(train_dir)
            test_path = os.path.dirname(test_dir)

            logging.info("creating directory for train and test")
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            logging.info("spliting the data into train and test")
            train, test = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)

            logging.info("loading the data to train.csv and test.csv")
            train.to_csv(train_dir, index=False, header=True)
            test.to_csv(test_dir, index=False, header=True)

            logging.info("completion of ingested directory")

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_dataframe()
            dataframe = self.export_to_feature_store(dataframe)
            self.export_to_ingested_dir(dataframe)
            dataIngestion = dataIngestionArtifact(
                train_dir= self.data_ingestion_config.train_store, 
                test_dir= self.data_ingestion_config.test_store
            )
            return dataIngestion

        except Exception as e:
            raise NetworkSecurityException(e, sys)


        



