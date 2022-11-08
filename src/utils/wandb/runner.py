from typing import Optional

from wandb.apis.public import Run

import wandb


class WandbArtifactRunner:
    """Class wrapper around Weights and Biases's runner management."""

    def run(
        wandb_entity: str,
        wandb_project: str,
        wandb_secret_key: str,
        wandb_experiment_id: str,
        wandb_experiment_name: str,
        wandb_experiment_description: Optional[str] = None,
        wandb_configs: Optional[str] = {},
    ) -> Run:
        """
        Method to initializate Weights and Biases runner.

        Parameters
        ----------
        wandb_entity : Entity that will log in Weights and Biases, can be user or group
            _description_
        wandb_project : Wandb project
            _description_
        wandb_secret_key : Wandb user secret key
            _description_
        wandb_experiment_name : Wandb experiment name
            _description_
        wandb_experiment_description : Optional[str], optional
            Wandb experiment description, by default None
        wandb_configs : Optional[str], optional
            Configurations to be logged in Wandb to help identify the experiment, by default {}

        Returns
        -------
        Run
            Weights and Biases Run session
        """

        wandb_run_name = f"{wandb_experiment_id}-{wandb_experiment_name}"
        wandb.login(key=wandb_secret_key, relogin=True)
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_run_name,
            notes=wandb_experiment_description,
            config=wandb_configs,
        )
        return run

    def run_v2(
        wandb_entity: str,
        wandb_project: str,
        wandb_user_name: str,
        wandb_secret_key: str,
        wandb_experiment_id: str,
        wandb_experiment_name: str,
        wandb_experiment_description: Optional[str] = None,
        wandb_configs: Optional[str] = {},
    ) -> Run:
        """
        _summary_

        Parameters
        ----------
        wandb_entity : str
            Entity that will log in Weights and Biases, can be user or group
        wandb_project : str
            Wandb project
        wandb_secret_key : str
            Wandb user secret key
        wandb_experiment_id :
            Wandb experiment id
        wandb_experiment_name :
            Wandb experiment name
        wandb_user_name : str
            Wandb user name
        wandb_experiment_description : Optional[str], optional
             Wandb experiment description, by default None
        wandb_configs : Optional[str], optional
            Configurations to be logged in Wandb to help identify the experiment, by default {}

        Returns
        -------
        Run
            _description_
        """
        wandb_run_name = f"{wandb_experiment_id}-{wandb_experiment_name}"
        wandb.login(key=wandb_secret_key, relogin=True)
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project if (wandb_project) else wandb_user_name,
            name=wandb_run_name,
            config=wandb_configs,
        )
        return run

    @staticmethod
    def finish_run(run: Run):
        """
        Method to terminate Weights and Biases Run session.

        Parameters
        ----------
        run : Run
            Weights and Biases Run session
        """
        run.finish()
