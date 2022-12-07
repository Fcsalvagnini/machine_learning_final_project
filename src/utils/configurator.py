from abc import ABCMeta, abstractclassmethod
from typing import Dict, List
import yaml

class Configurations(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self) -> None:
        pass

    def setattrs(self, configurations: Dict) -> None:
        for config_name, config_value in configurations.items():
            if hasattr(self, config_name):
                # Configures nested objects
                if isinstance(getattr(self, config_name), Configurations):
                    nested_configuration = getattr(self, config_name)
                    nested_configuration.setattrs(
                        configurations=configurations[config_name]
                    )
                    continue
                setattr(self, config_name, config_value)

    def log(self, logger, parent_attr=""):
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), Configurations):
                nested_class = getattr(self, attr)
                nested_class.log(logger, parent_attr=f"[{attr}]")
                continue

            log_message = ""
            log_message += parent_attr
            log_message += f"[{attr}]: {getattr(self, attr)}"
            logger.info(log_message)

class AugmentationsConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.augmentations: Dict = {}

        self.setattrs(configurations=configurations)

class DatasetConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.data_path: str = ""
        self.data_descriptors_path: str = ""
        self.phase: str = ""
        self.n_modalities: int = 2
        self.batch_size: int = 2
        self.num_threads: int = 4
        self.device_id: int = 0
        self.patch_size: list = [128, 128, 128]
        self.crop: bool = True
        self.evaluate: bool = False

        self.setattrs(configurations=configurations)

class LayersConfigurations(Configurations, metaclass=ABCMeta):
    def __init__(self, configurations: Dict) -> None:
        self.block_names = []
        self.block_parameters = []

    def setattrs(self, configurations: Dict) -> None:
        for name, parameters in configurations.items():
            self.block_names.append(name)
            self.block_parameters.append(parameters)

class DataLoaderConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.augment: bool = False
        self.augmentations: List = []
        self.patch_training: bool = False

        self.dataset: Configurations = DataConfigs(configurations = {})

        self.setattrs(configurations=configurations)

class TrainConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.model_tag: str = ""
        self.checkpoints_path: str = ""
        self.deep_supervision: bool = False
        self.epochs: int = 100
        self.batch_size: int = 8
        self.loss: dict = {}
        self.optimizer: dict = {}
        self.scheduler: Configurations = SchedulerConfigs(configurations={})
        self.logging_level: str = "INFO"
        self.gpu_id: int = 0

        self.data_loader: Configurations = DataLoaderConfigs(configurations={})

        self.setattrs(configurations=configurations)

class ValidationConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.model_tag: str = ""
        self.checkpoint_path: str = ""
        self.data_path: str = ""

        self.setattrs(configurations=configurations)

class DataConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.data_paths: list = []
        self.data_descriptors: list = []

        self.setattrs(configurations=configurations)

class SchedulerConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.scheduler_fn: str = ""
        self.from_monai: bool = False
        self.scheduler_kwargs: Dict = {}

        self.setattrs(configurations=configurations)

class ModelConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.depth: int = 3
        self.encoder: LayersConfigurations = LayersConfigurations(configurations={})
        self.decoder: LayersConfigurations = LayersConfigurations(configurations={})
        self.skip_connections: LayersConfigurations = LayersConfigurations(configurations={})
        self.deep_supervision: bool = False

        self.setattrs(configurations=configurations)


class WandbInfo(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.wandb_entity: str
        self.wandb_project: str
        self.wandb_secret_key: str

        self.setattrs(configurations=configurations)


if __name__ == "__main__":
    # Parses the experiment configurations
    with open("../experiment_configs/sample_config.yaml") as yaml_file:
        experiment_configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

    train_configs = TrainConfigs(experiment_configs["train_configs"])

    print(train_configs.epochs)
    print(train_configs.batch_size)
    print(train_configs.loss)
    print(train_configs.optimizer)
    print(train_configs.data_loader.augment)
    print(train_configs.data_loader.augmentations)
    print(train_configs.data_loader.patch_training)
