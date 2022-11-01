from typing import Dict
import torch
import logging
import argparse
import sys
import yaml
from torchsummary import summary
from tqdm import trange
import os
import gc

from src.utils.global_vars import LOGGING_LEVEL, LOSSES, OPTIMIZERS
from src.utils.configurator import TrainConfigs, DatasetConfigs
from src.models.segmentation_architecture import get_model
from src.datasets.dataset import BrainDataset
from src.utils.schedulers import get_scheduler
from src.utils.callbacks import SaveBestModel

from torch.utils.data import DataLoader

def run_train_epoch(model, optimizer, loss, dataloader, monitoring_metrics,
                epoch):
    model.train()
    model.to("cuda")
    runnin_loss = 0

    with trange(len(dataloader), desc="Train Loop") as progress_bar:
        for batch_idx, batch in zip(progress_bar, dataloader):
            volumetric_image, segmentation_mask = [
                data.to("cuda") for data in batch
            ]
            optimizer.zero_grad()
            predicted_segmentation = model(volumetric_image)
            batch_loss = loss(predicted_segmentation, segmentation_mask)
            batch_loss.backward()
            optimizer.step()

            running_loss += loss.cpu()

            progress_bar.set_postfix(
                desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f}"
            )

    epoch_loss = (runnin_loss / len(dataloader)).detach().numpy()
    monitoring_metrics["loss"]["train"].append(epoch_loss)

    return epoch_loss

def run_validation_epcoch(model, optimizer, loss, dataloader, monitoring_metrics,
                epoch, configurations, save_best_model):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to("cuda")
        model.eval()
        running_loss = 0

        with trange(len(dataloader), desc="Validation Loop") as progress_bar:
            for batch_idx, batch in zip(progress_bar, dataloader):
                volumetric_image, segmentation_mask = [
                    data.to("cuda") for data in batch
                ]
                predicted_segmentation = model(volumetric_image)
                batch_loss = loss(predicted_segmentation, segmentation_mask)

                running_loss += batch_loss.cpu()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f}"
                )

    epoch_loss = (running_loss / len(dataloader)).detach().numpy()
    monitoring_metrics["loss"]["validation"].append(epoch_loss)

    save_best_model(epoch_loss, epoch, model, optimizer, loss, configurations)

    return epoch_loss

def train_loop(model, train_dataloader, validation_dataloader, optmizer, loss,
            scheduler, train_configs):
    os.makedirs(train_configs.checkpoints_path, exist_ok=True)
    monitoring_metrics = {
        "loss": {"train": [], "validation": []}
    }
    save_best_model = SaveBestModel()

    for epoch in range(1, train_configs.epochs + 1):
        train_loss = run_train_epoch(
            model, optmizer, loss, train_dataloader, monitoring_metrics,
            epoch
        )
        valid_loss = run_validation_epcoch(
            model, optmizer, loss, validation_dataloader, monitoring_metrics,
            epoch, train_configs, train_configs, save_best_model
        )
        scheduler.step(monitoring_metrics["loss"]["validation"][-1])

def train(configs: Dict) -> None:
    train_configs = TrainConfigs(configs["train_configs"])
    train_dataset_configs = DatasetConfigs(configs["train_configs"]["data_loader"]["dataset"])
    train_dataset = BrainDataset(
        cfg=train_dataset_configs,
        phase="train",
        num_concat=2
    )
    validation_dataset = BrainDataset(
        cfg=train_dataset_configs,
        phase="validation",
        num_concat=2
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_configs.batch_size
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=train_configs.batch_size
    )

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
    logger.info(
        f"Started Training Experiment with Model Architecture:"
    )
    summary(
        model, input_size=(2, 128, 128, 128),
        batch_size=train_configs.batch_size, show_input=True,
        show_hierarchical=True
    )

    loss = LOSSES[train_configs.loss["name"]](**train_configs.loss["parameters"])
    optmizer = OPTIMIZERS[train_configs.optimizer["name"]](
        model.parameters(), **train_configs.optimizer["parameters"]
    )
    scheduler = get_scheduler(
        train_configs.scheduler.scheduler_fn,
        optimizer=optmizer, from_monai=train_configs.scheduler.from_monai,
        **train_configs.scheduler.scheduler_kwargs
    )

    train_loop(
        model, train_dataloader, validation_dataloader, optmizer, loss,
        scheduler, train_configs
    )

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