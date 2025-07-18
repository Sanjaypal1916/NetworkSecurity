from network_sec.components.data_ingestion import DataIngestion
from network_sec.exception.exception import NetworkSecurityException
from network_sec.logs.logger import logging
from network_sec.entity.config_entity import trainingpipeline, dataIngestionConfig, dataValidationConfig
import pymongo

from network_sec.components.data_ingestion import DataIngestion
from network_sec.components.data_validation import dataValidation
from network_sec.components.data_transformation import DataTransformation
from network_sec.exception.exception import NetworkSecurityException
from network_sec.logs.logger import logging
from network_sec.entity.config_entity import dataIngestionConfig,dataValidationConfig, dataTransformationConfig
from network_sec.entity.config_entity import ModelTrainerConfig

from network_sec.components.model_training import ModelTrainer
from network_sec.entity.config_entity import ModelTrainerConfig
 


import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=trainingpipeline()
        dataingestionconfig=dataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)

        # record = pymongo.MongoClient("mongodb+srv://sanjaypal606060:admin123@cluster0.locouzx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        # # print(record)
        # collection = record["NetworkSecurity"]["NetworkData"]
        # data = list(collection.find())
        # print(data)

        data_validation_config=dataValidationConfig(trainingpipelineconfig)
        data_validation=dataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_dataingestion()
        logging.info("data Validation Completed")
        print(data_validation_artifact)

        data_transformation_config=dataTransformationConfig(trainingpipelineconfig)
        dataTransformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("initiated data transformation")
        data_tranformation_artifact=dataTransformation.initiate_data_transformation()
        logging.info("data transformation completed")
        print(data_tranformation_artifact)

        logging.info("Model Training sstared")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_tranformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")
        
        
        
    except Exception as e:
           raise NetworkSecurityException(e,sys)
