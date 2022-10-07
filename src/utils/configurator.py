from typing import Dict, List
from xmlrpc.client import Boolean
import yaml
from abc import ABCMeta, abstractclassmethod

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

class DataLoaderConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.augment: Boolean = False
        self.augmentations: List = []
        self.patch_training: Boolean = False

        self.setattrs(configurations=configurations)

class TrainConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.epochs: int = 100
        self.batch_size: int = 8
        self.loss: str = "RMSE"
        self.optimizer: str = "SGD"

        self.data_loader: Configurations = DataLoaderConfigs(configurations={})

        self.setattrs(configurations=configurations)

class DataConfigs(Configurations):
    def __init__(self, configurations: Dict) -> None:
        self.data_paths:list = []
        self.data_descriptors:list = []

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
