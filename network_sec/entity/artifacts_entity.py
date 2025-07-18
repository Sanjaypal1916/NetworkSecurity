from dataclasses import dataclass
from pathlib import Path

@dataclass
class dataIngestionArtifact:
    train_dir: str
    test_dir: str

@dataclass
class DataValidationArtifact:
   
    validation_status : bool
    valid_train_file_path : Path
    valid_test_file_path : Path
    invalid_train_file_path : Path
    invalid_test_file_path : Path
    drift_report_file_path : Path

@dataclass
class dataTransformationArtifact:
    object_file_path : Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact