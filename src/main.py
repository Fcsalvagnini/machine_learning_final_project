from ast import arg
import torch
import logging
import argparse
import sys
import yaml

from utils.global_vars import LOGGING_LEVEL
from utils.configurator import TrainConfigs

def train(configs):        
    train_configs = TrainConfigs(configs["train_configs"])
    
    logging.basicConfig(
        stream=sys.stdout, 
        level=LOGGING_LEVEL[train_configs.logging_level]
    )
    logger = logging.getLogger(name="TRAIN")
    logger.info(
        f"Starting Training Experiment with model {train_configs.model}"
    )
    train_configs.log(logger)

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