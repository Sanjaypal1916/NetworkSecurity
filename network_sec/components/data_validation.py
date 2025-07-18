


from network_sec.exception.exception import NetworkSecurityException
from network_sec.logs.logger import logging


## configuration of the Data Ingestion Config

from network_sec.entity.config_entity import dataIngestionConfig, dataValidationConfig
from network_sec.entity.artifacts_entity import dataIngestionArtifact
from network_sec.constants.allconstants import *
import os
import sys
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from network_sec.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
from network_sec.entity.artifacts_entity import DataValidationArtifact
load_dotenv()


class dataValidation():
    def __init__(self, data_ingestion_artifact:dataIngestionArtifact,
                 data_validation_config:dataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(filepath) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def number_of_columns(self, dataframe : pd.DataFrame) -> bool:
        try:
            num_of_columns = len(self._schema_config["columns"])
            num_of_newcols = len(dataframe.columns)
            if num_of_newcols == num_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(s, sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

        except Exception as e:
            raise NetworkSecurityException(e,sys)

        

        
    def initiate_dataingestion(self):
        try:
            train_path = self.data_ingestion_artifact.train_dir
            test_path = self.data_ingestion_artifact.test_dir

            train_data = dataValidation.read_data(train_path)
            test_data = dataValidation.read_data(test_path)

            status = self.number_of_columns(train_data)
            if not status:
                error_message=f"Train dataframe does not contain all columns.\n"
            status = self.number_of_columns(dataframe=test_data)
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"

            ## lets check datadrift
            status=self.detect_dataset_drift(base_df=train_data,current_df=test_data)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_data.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_data.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.train_dir,
                valid_test_file_path=self.data_ingestion_artifact.test_dir,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)


