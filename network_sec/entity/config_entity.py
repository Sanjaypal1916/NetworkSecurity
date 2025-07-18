

from datetime import datetime
import os
from network_sec.constants import allconstants

print(allconstants.PIPELINE_NAME)
print(allconstants.ARTIFACT_DIR)



class trainingpipeline:
    def __init__(self, timestamp = datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = allconstants.PIPELINE_NAME
        self.artifacts_name = allconstants.ARTIFACT_DIR
        self.artifacts_dir = os.path.join(self.artifacts_name, timestamp)
        self.model_name = os.path.join("final_model")
        self.timestamp : str = timestamp

class dataIngestionConfig():
    def __init__(self, trainingpipeline : trainingpipeline):
        self.data_ingestion_dir = os.path.join(trainingpipeline.artifacts_dir, allconstants.DATA_INGESTION_DIR_NAME )
        self.ingested = os.path.join(self.data_ingestion_dir, allconstants.DATA_INGESTION_INGESTED_DIR, allconstants.FILE_NAME)
        self.feature_store_dir = os.path.join(self.data_ingestion_dir, allconstants.DATA_INGESTION_FEATURE_STORE_DIR, allconstants.FILE_NAME)
        self.train_store = os.path.join(self.data_ingestion_dir, allconstants.DATA_INGESTION_FEATURE_STORE_DIR, allconstants.TRAIN_FILE_NAME)
        self.test_store = os.path.join(self.data_ingestion_dir, allconstants.DATA_INGESTION_FEATURE_STORE_DIR, allconstants.TEST_FILE_NAME)
        self.collection : str = os.path.join(allconstants.DATA_INGESTION_COLLECTION_NAME)
        self.database : str = os.path.join(allconstants.DATA_INGESTION_DATABASE_NAME)
        self.train_test_split_ratio : float= allconstants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


class dataValidationConfig():
    def __init__(self, trainingpipeline : trainingpipeline):
        self.data_validation_dir = os.path.join(trainingpipeline.artifacts_dir, allconstants.DATA_VALIDATION_DATA_DIR)
        self.validate = os.path.join(self.data_validation_dir, allconstants.DATA_VALIDATION_VALID)
        self.invalide = os.path.join(self.data_validation_dir, allconstants.DATA_VALIDATION_INVALID)
        self.valid_train_file_path: str = os.path.join(self.validate, allconstants.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.validate, allconstants.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalide, allconstants.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalide, allconstants.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            allconstants.DATA_VALIDATION_DATA_DRIFT_REP,
            allconstants.DATA_VALIDATION_DATA_DRIFT_DIR,
        )


class dataTransformationConfig():
    def __init__(self, trainingpipeline : trainingpipeline):
        self.data_transformation_dir: str = os.path.join( trainingpipeline.artifacts_dir,allconstants.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,allconstants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,allconstants.TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  allconstants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,allconstants.TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, allconstants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,allconstants.PREPROCESSING_OBJECT_FILE_NAME,)
        



        
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:trainingpipeline):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifacts_dir, allconstants.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, allconstants.MODEL_TRAINER_TRAINED_MODEL_DIR, allconstants.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = allconstants.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = allconstants.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
