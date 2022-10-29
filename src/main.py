from typing import Dict
import torch
import logging
import argparse
import sys
import yaml

from src.utils.global_vars import LOGGING_LEVEL
from src.utils.configurator import DataConfigs, TrainConfigs, DatasetConfigs
from src.models.segmentation_architecture import get_model
from src.datasets.dataset import BrainDataset

from torch.utils.data import DataLoader
import numpy as np

def train(configs: Dict) -> None:
    train_configs = TrainConfigs(configs["train_configs"])
    train_dataset_configs = DatasetConfigs(configs["train_configs"]["data_loader"]["dataset"])
    train_dataset = BrainDataset(
        cfg=train_dataset_configs,
        phase="train",
        num_concat=2
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    
    logging.basicConfig(
        stream=sys.stdout,
        level=LOGGING_LEVEL[train_configs.logging_level]
    )
    logger = logging.getLogger(name="TRAIN")
    logger.info(
        f"Starting Training Experiment with model {train_configs.model_tag}"
    )
    train_configs.log(logger)

    model = get_model(configs)

    array = torch.Tensor(np.ones((4, 2, 128, 128, 128)))
    array.to("cuda")
    model.to("cuda")

    y = model(array)
    print(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Glioma Segmentation Framework",
        description="Training Framework for experimentation with segmentation "\
                    "models towards Glioma Delineation"
    )
    parser.add_argument("mode", help="Execution mode (train or test)")
    parser.add_argument("config_file", help="Path to configuration file (YAML)")
    args = parser.parse_args()

    with open(args.config_file) as yaml_file:
        configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if args.mode == "train":
        train(configs)