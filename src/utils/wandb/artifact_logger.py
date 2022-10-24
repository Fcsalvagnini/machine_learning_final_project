import tempfile
from pathlib import Path
from typing import Dict

import tensorflow as tf
from wandb.apis.public import Run

import wandb
from src.utils.zip.zip import ZipfileHandler


class WandbArtifactLogger:
    """This class is a wrapper to log artifacts to Weights and Biases."""

    @staticmethod
    def log_data(run: Run, data: Path, artifact_name: str, finish_run: bool = False) -> None:

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

        tempdir = tempfile.mkdtemp()
        # ZipfileHandler.unzip_from_bentoml_file(file=data, output_dirpath=tempdir)
        ZipfileHandler.unzip(zip_path=data, output_dirpath=tempdir)
        WandbArtifactLogger.__log_artifact(run=run, artifact_name=artifact_name, artifact_type="data", source_dirpath=tempdir)

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
        WandbArtifactLogger.__log_artifact(
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
        WandbArtifactLogger.__log_artifact(run=run, artifact_name=artifact_name, artifact_type="model", source_dirpath=tempdir)

    @staticmethod
    def __log_artifact(run: Run, artifact_name: str, artifact_type: str, source_dirpath: str, artifact_metadata: Dict = None) -> None:
        """
        Private method to log Weights and Biases's Artifacts.

        Parameters
        ----------
        run : Run
            Weights and Biases run session
        artifact_name : str
            Name of the artifact in Weights and Biases
        artifact_type : str
            Type of the artifact in Weights and Biases
        source_dirpath : str
            Source diretory of the artifact
        artifact_metadata: Dict
            Artifact metadata
        """

        artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=artifact_metadata)
        # as for the model, since ModelCheckpoint is saving in the tf-saved model format, so
        # add_dir won't show error
        artifact.add_dir(source_dirpath)
        run.log_artifact(artifact)

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
        WandbArtifactLogger.__log_artifact(run=run, artifact_name=artifact_name, artifact_type=artifact_type, source_dirpath=source_dirpath)
