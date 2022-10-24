import tempfile
from pathlib import Path
from typing import Dict
import wandb

import tensorflow as tf
from wandb.apis.public import Run

from src.utils.wandb.logger import WandbLogger

class WandbArtifactLogger:
    """This class is a wrapper to log artifacts to Weights and Biases."""

    @staticmethod
    def log_data_from_zip(run: Run, data: Path, artifact_name: str, finish_run: bool = False) -> None:

        """
        Method to log data to Weights and Biases.

        Parameters
        ----------
        run : Run
            Weights and Biases run session
        data : Path
            Absolute path of compressed dataset. Supported format: 'zip'
        artifact_name : str
            Name of the artifact in Weights and Biases
        finish_run : bool, optional
            Finish run after logging, by default False
        """

        #tempdir = tempfile.mkdtemp()
        #ZipfileHandler.unzip(zip_path=data, output_dirpath=tempdir)
        #WandbArtifactLogger.__log_artifact(run=run, artifact_name=artifact_name, artifact_type="data", source_dirpath=tempdir)
        raise NotImplementedError("Method to log data from zip not implemented")

    @staticmethod
    def log_data_from_dir(run: Run, dir_path: str, artifact_name: str, artifact_metadata: dict):
        """
        Method to log dataset from directory into Weights and Biases.

        Parameters
        ----------
        run : Run
            Weights and Biases run session
        dir_path : str
            Dataset's diretory absolute path
        artifact_name : str
            Name of the artifact in Weights and Biases
        artifact_metadata : dict
            Artifact metadata
        """
        WandbLogger.log_artifact(
            run=run,
            artifact_name=artifact_name,
            artifact_type="data",
            source_dirpath=dir_path,
            artifact_metadata=artifact_metadata,
        )

    @staticmethod
    def log_model(run: Run, model: tf.keras.models.Model, artifact_name: str) -> None:
        """
        Method to log tensorflow models to Weights and Biases.

        Parameters
        ----------
        run : Run
            Weights and Biases run session
        model : Model
            Tensorflow model to be logged
        artifact_name : str
            Name of the artifact in Weights and Biases
        finish_run : bool, optional
            Finish run after logging, by default False
        """

        tempdir = tempfile.mkdtemp()
        tf.saved_model.save(model, tempdir)
        WandbLogger.log_artifact(run=run, artifact_name=artifact_name, artifact_type="model", source_dirpath=tempdir)



    @staticmethod
    def log_saved_model_artifact(run: Run, artifact_name: str, artifact_type: str, source_dirpath: str) -> None:
        """
        log a tf-saved model format artifact to wandb for a given run.

        Parameters
        ----------
        run : Run
            Weights and Biases run session
        artifact_name : str
            name of the model that the user selected at time of training
        artifact_type : str
            must be 'model'
        source_dirpath : str
            path of the dir. containing model in tf-saved model format
        """
        WandbLogger.log_artifact(run=run, artifact_name=artifact_name, artifact_type=artifact_type, source_dirpath=source_dirpath)
