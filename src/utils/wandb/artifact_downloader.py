from pathlib import Path
from typing import Dict, Optional

import wandb


class WandbArtifactDownloader:
    """Class wrapper around wandb's ArtifactDownloader."""

    @staticmethod
    def download_artifact(
        wandb_entity: str,
        wandb_project: str,
        wandb_secret_key: str,
        artifact_name: str,
        artifact_version: str,
        local_download_folder: Optional[Path] = None,
    ):
        """
        Method to download Artifacts from Weights and Biases.

        Parameters
        ----------
        wandb_entity : str
            Weights and Biases entity name
        wandb_project : str
            Weights and Biases project name
        wandb_secret_key : str
            Weights and Biases secret key
        artifact_name : str
             Artificat name in Weights and Biases
        artifact_version : str
            Version of artifact in Weights and Biases
        local_download_folder : Path, optional
            _description_, by default None

        Returns
        -------
        Path
            Path to the downloaded artifact
        """

        wandb.login(key=wandb_secret_key, relogin=True)
        api = wandb.Api()
        artifact = api.artifact(f"{wandb_entity}/{wandb_project}/{artifact_name}:{artifact_version}")
        local_download_folder = str(local_download_folder) if local_download_folder else "artifacts"
        artifact.download(local_download_folder)

        return local_download_folder

    @staticmethod
    def get_metadata(wandb_entity: str, wandb_project: str, wandb_secret_key: str, artifact_name: str, artifact_version: str) -> Dict:
        """
        Method to get Artifacts Metadata from Weights and Biases.

        Parameters
        ----------
        wandb_entity : str
            Weights and Biases entity name
        wandb_project : str
            Weights and Biases project name
        wandb_secret_key : str
            Weights and Biases secret key
        artifact_name : str
             Artificat name in Weights and Biases
        artifact_version : str
            Version of artifact in Weights and Biases

        Returns
        -------
        artifact_metadata: Dict
            Artifact Metadata
        """

        wandb.login(key=wandb_secret_key, relogin=True)
        api = wandb.Api()
        artifact = api.artifact(f"{wandb_entity}/{wandb_project}/{artifact_name}:{artifact_version}")
        artifact_metadata = artifact.metadata
        return artifact_metadata
