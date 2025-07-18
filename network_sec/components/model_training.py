import os
import sys
import optuna
from optuna.pruners import MedianPruner

from network_sec.exception.exception import NetworkSecurityException
from network_sec.logs.logger import logging

from network_sec.entity.artifacts_entity import dataTransformationArtifact, ModelTrainerArtifact
from network_sec.entity.config_entity import ModelTrainerConfig

from network_sec.utils.ml_utils.model import NetworkModel
from network_sec.utils import save_object, load_object
from network_sec.utils import load_numpy_array_data
from network_sec.utils.ml_utils.metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import r2_score

import mlflow
from urllib.parse import urlparse
import dagshub

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/krishnaik06/networksecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "sanjaypal1916"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "43d0fdb0ba74e9aaf38d5bc3e190aac3933b13db"

dagshub.init(repo_owner='sanjaypal1916', repo_name='NetworkSecurity', mlflow=True)


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: dataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

   

    def track_mlflow(self, best_model, classificationmetric, X_test=None, y_test=None):
        mlflow.set_tracking_uri("https://dagshub.com/sanjaypal1916/NetworkSecurity.mlflow")

        with mlflow.start_run():
            # ✅ Log basic metrics
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)

            # ✅ Log r2 score if available (for regression-like metrics)
            if hasattr(classificationmetric, "r2_score"):
                mlflow.log_metric("r2_score", classificationmetric.r2_score)

            # ✅ Log model name as a parameter
            mlflow.log_param("model_name", best_model.__class__.__name__)
            params_to_log = {k: v for k, v in best_model.get_params().items() if isinstance(v, (int, float, str, bool))}
            mlflow.log_params(params_to_log)
            
            model_path = self.model_trainer_config.trained_model_file_path
            mlflow.log_artifact(model_path, artifact_path="models")
            # ✅ Log the trained model as an artifact (safe for Dagshub)
            try:
                mlflow.sklearn.log_model(best_model, "model")
            except Exception as e:
                print(f"[Warning] Dagshub does not support model registry, logged as artifact only: {e}")



    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting Optuna hyperparameter tuning...")

            def objective(trial):
                # ✅ Model Selection
                model_name = trial.suggest_categorical("model", [
                    "Random Forest", "Decision Tree", "Gradient Boosting", "Logistic Regression", "AdaBoost"
                ])

                if model_name == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_categorical("rf_n_estimators", [8, 16, 32, 128, 256]),
                        criterion=trial.suggest_categorical("rf_criterion", ["gini", "entropy", "log_loss"]),
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(
                        criterion=trial.suggest_categorical("dt_criterion", ["gini", "entropy", "log_loss"]),
                        random_state=42
                    )
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        learning_rate=trial.suggest_float("gb_lr", 0.001, 0.1, log=True),
                        subsample=trial.suggest_float("gb_subsample", 0.6, 0.9),
                        n_estimators=trial.suggest_categorical("gb_n_estimators", [8, 16, 32, 64, 128, 256]),
                        random_state=42
                    )
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
                elif model_name == "AdaBoost":
                    model = AdaBoostClassifier(
                        learning_rate=trial.suggest_float("ada_lr", 0.001, 0.1, log=True),
                        n_estimators=trial.suggest_categorical("ada_n_estimators", [8, 16, 32, 64, 128, 256]),
                        random_state=42
                    )

                # ✅ Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # ✅ Use R2 Score as objective
                score = r2_score(y_test, y_pred)

                # ✅ Early prune bad trials
                if score < 0.5:
                    raise optuna.TrialPruned()

                return score

            study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))
            study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=True)

            best_params = study.best_params
            best_model_name = best_params["model"]

            logging.info(f"✅ Best Model: {best_model_name} | R2 Score: {study.best_value}")
            logging.info(f"✅ Best Params: {best_params}")

            # ✅ Recreate Best Model
            if best_model_name == "Random Forest":
                best_model = RandomForestClassifier(
                    n_estimators=best_params["rf_n_estimators"],
                    criterion=best_params["rf_criterion"],
                    random_state=42,
                    n_jobs=-1
                )
            elif best_model_name == "Decision Tree":
                best_model = DecisionTreeClassifier(
                    criterion=best_params["dt_criterion"],
                    random_state=42
                )
            elif best_model_name == "Gradient Boosting":
                best_model = GradientBoostingClassifier(
                    learning_rate=best_params["gb_lr"],
                    subsample=best_params["gb_subsample"],
                    n_estimators=best_params["gb_n_estimators"],
                    random_state=42
                )
            elif best_model_name == "Logistic Regression":
                best_model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
            elif best_model_name == "AdaBoost":
                best_model = AdaBoostClassifier(
                    learning_rate=best_params["ada_lr"],
                    n_estimators=best_params["ada_n_estimators"],
                    random_state=42
                )

            # ✅ Retrain Best Model on full training data
            best_model.fit(X_train, y_train)

            # ✅ Track metrics in MLflow
            y_train_pred = best_model.predict(X_train)
            train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            

            y_test_pred = best_model.predict(X_test)
            test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # ✅ Save Best Model
            preprocessor = load_object(file_path=self.data_transformation_artifact.object_file_path)
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            save_object("final_model/model.pkl", best_model)
            save_object("final_model/preprocessor.pkl", preprocessor)

            self.track_mlflow(best_model, train_metrics)
            self.track_mlflow(best_model, test_metrics)
            
            # ✅ Return Artifact
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
