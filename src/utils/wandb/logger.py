
import wandb

from typing import Dict
from wandb.apis.public import Run


class WandbLogger:

    @staticmethod
    def log_artifact(
        run: Run, 
        artifact_name: str, 
        artifact_type: str, 
        source_dirpath: str, 
        artifact_metadata: Dict = None
        ) -> None:
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