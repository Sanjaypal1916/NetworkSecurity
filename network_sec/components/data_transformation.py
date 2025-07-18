
from network_sec.entity.config_entity import dataTransformationConfig
from network_sec.entity.artifacts_entity import DataValidationArtifact

from network_sec.exception.exception import NetworkSecurityException
from network_sec.logs.logger import logging
from network_sec.entity.config_entity import dataIngestionConfig, dataValidationConfig
from network_sec.constants.allconstants import *
import os
import sys
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import List
from network_sec.constants import allconstants
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from network_sec.utils import read_yaml_file, write_yaml_file, save_numpy_array_data, save_object
from scipy.stats import ks_2samp
from network_sec.entity.artifacts_entity import DataValidationArtifact, dataTransformationArtifact

load_dotenv()

class DataTransformation:
    def __init__(self, dataValidationArtifact: DataValidationArtifact, datatransform: dataTransformationConfig):
        try:
            self.dataValidat = dataValidationArtifact
            self.datatransform = datatransform
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read(filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def get_data_transformer_object(self) -> Pipeline:
        try:
            imputer = KNNImputer(**allconstants.DATA_TRANSFORMATION_IMPUTER_PARAMS)
            pipeline = Pipeline([("imputer", imputer)])
            logging.info("Created Pipeline using KNNImputer")
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(self) -> dataTransformationArtifact:
        try:
            train_path = self.dataValidat.valid_train_file_path
            test_path = self.dataValidat.valid_test_file_path

            train_data = self.read(train_path)
            test_data = self.read(test_path)

            preprocessor = self.get_data_transformer_object()

            input_train_data = train_data.drop([allconstants.TARGET_COLUMN], axis=1)
            input_test_data = test_data.drop([allconstants.TARGET_COLUMN], axis=1)
            output_train_data = train_data[allconstants.TARGET_COLUMN]
            output_test_data = test_data[allconstants.TARGET_COLUMN]

            train_transformed_data = preprocessor.fit_transform(input_train_data)
            test_transformed_data = preprocessor.transform(input_test_data)

            train_arr = np.c_[train_transformed_data, np.array(output_train_data)]
            test_arr = np.c_[test_transformed_data, np.array(output_test_data)]

            save_numpy_array_data(self.datatransform.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.datatransform.transformed_test_file_path, test_arr)

            save_object(self.datatransform.transformed_object_file_path, obj=preprocessor)

            data_transformation_artifact = dataTransformationArtifact(
                object_file_path=self.datatransform.transformed_object_file_path,
                transformed_train_file_path=self.datatransform.transformed_train_file_path,
                transformed_test_file_path=self.datatransform.transformed_test_file_path
            )

            logging.info("Data Transformation Completed Successfully")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
