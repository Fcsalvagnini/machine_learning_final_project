from typing import Dict
import torch
import logging
import argparse
import numpy as np
import sys
import csv
import yaml
from torchsummary import summary
from tqdm import trange, tqdm
import os
import gc
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
#from wandb.apis.public import Run

import nibabel as nib

from src.utils.global_vars import LOGGING_LEVEL, LOSSES, OPTIMIZERS
from src.utils.configurator import TrainConfigs, DatasetConfigs, ValidationConfigs, WandbInfo
from src.models.segmentation_architecture import get_model
from src.datasets.dataset import BrainDataset
from src.utils.schedulers import get_scheduler
from src.utils.callbacks import SaveBestModel
from src.metrics.dice import DiceMetric
from src.utils.csv.csv_writter import CsvWritter
from src.dali_dataloader.pipelines import DaliFullPipeline

#from src.utils.wandb.runner import WandbArtifactRunner
#from src.utils.wandb.logger import WandbLogger

from torch.utils.data import DataLoader

def run_train_epoch(model, optimizer, scheduler, loss, dice_metric, pipeline,
        monitoring_metrics, epoch
    ):
    model.train()
    model.to("cuda")
    running_loss = 0
    running_dice = 0

    steps = int(
        np.ceil(pipeline.nift_iterator.dataset_len / pipeline.batch_size)
    )

    with trange(steps, desc="Train Loop") as progress_bar:
        for batch_idx in progress_bar:
            volumetric_images, segmentation_masks = pipeline.run()
            volumetric_images = torch.Tensor(
                volumetric_images.as_cpu().as_array()
            ).to("cuda")
            segmentation_masks = torch.Tensor(
                segmentation_masks.as_cpu().as_array()
            ).to("cuda")
            optimizer.zero_grad()
            predicted_segmentation = model(volumetric_images)
            
            batch_loss = loss.forward(predicted_segmentation, segmentation_masks)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.cpu()
            running_dice += torch.mean(dice_metric.compute(predicted_segmentation, segmentation_masks)).cpu()

            progress_bar.set_postfix(
                desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f} | Dice: {running_dice / (batch_idx + 1):.3f}"
            )

            scheduler.step()

    epoch_loss = (running_loss / steps).detach().numpy()
    epoch_dice = (running_dice / steps).detach().numpy()
    monitoring_metrics["loss"]["train"].append(epoch_loss)
    monitoring_metrics["dice"]["train"].append(epoch_dice)

def run_validation_epoch(model, loss, dice_metric, pipeline,
                monitoring_metrics, epoch, configurations, 
                save_best_model):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to("cuda")
        model.eval()
        running_loss = 0
        running_dice = 0

        steps = int(
            np.ceil(pipeline.nift_iterator.dataset_len / pipeline.batch_size)
        )

        with trange(steps, desc="Validation Loop") as progress_bar:
            for batch_idx in progress_bar:
                volumetric_images, segmentation_masks = pipeline.run()
                volumetric_images = torch.Tensor(
                    volumetric_images.as_cpu().as_array()
                ).to("cuda")
                segmentation_masks = torch.Tensor(
                    segmentation_masks.as_cpu().as_array()
                ).to("cuda")
                predicted_segmentation = model(volumetric_images)
                batch_loss = loss.forward(predicted_segmentation, segmentation_masks)

                running_loss += batch_loss.cpu()
                running_dice += torch.mean(
                    dice_metric.compute(
                        predicted_segmentation, segmentation_masks
                    )
                ).cpu()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f} | Dice: {running_dice / (batch_idx + 1):.3f}"
                )

    epoch_loss = (running_loss / steps).detach().numpy()
    epoch_dice = (running_dice / steps).detach().numpy()
    monitoring_metrics["loss"]["validation"].append(epoch_loss)
    monitoring_metrics["dice"]["validation"].append(epoch_dice)

    save_best_model(epoch_loss, epoch, model, configurations)

def train_loop(model, train_pipeline, validation_pipeline, optmizer, loss,
            dice_metric, scheduler, train_configs):
    os.makedirs(train_configs.checkpoints_path, exist_ok=True)
    monitoring_metrics = {
        "loss": {"train": [], "validation": []},
        "dice": {"train": [], "validation": []}
    }
    save_best_model = SaveBestModel()
    csv_path = os.path.join(train_configs.checkpoints_path, "train_history.csv")
    csv_writer = CsvWritter(
        path=csv_path,
        header=[
            "epoch", "train_loss", "train_dice", "validation_loss",
            "validation_dice"
        ]
    )

    for epoch in range(1, train_configs.epochs + 1):
        run_train_epoch(
            model, optmizer, scheduler, loss, dice_metric, train_pipeline, monitoring_metrics,
            epoch
        )
        train_pipeline.reset()
        run_validation_epoch(
            model, loss, dice_metric, validation_pipeline, monitoring_metrics,
            epoch, train_configs, save_best_model
        )
        validation_pipeline.reset()
        csv_writer.write_line(
            content=[
                epoch, monitoring_metrics["loss"]["train"][-1],
                monitoring_metrics["dice"]["train"][-1],
                monitoring_metrics["loss"]["validation"][-1],
                monitoring_metrics["dice"]["validation"][-1]
                ]
        )

    # [ ] - run.log or wandb.log
    #run.log(
    #    monitoring_metrics
    #)

def train(configs: Dict) -> None:
    train_configs = TrainConfigs(configs["train_configs"])
    torch.cuda.set_device(train_configs.gpu_id)
    #wandb_info = WandbInfo(train_configs["wandb_info"])
    
    #wandb_info.update({
    #    "wandb_experiment_id": 1,
    #    "wandb_experiment_name": train_configs["model_tag"]
    #})

    #run = WandbArtifactRunner.run(**wandb_info)
    """
    TODO:
        [ ] -  Get lattest wandb_experiment_id with same wandb_experiment_name
        [ ] -  
    """
    train_pipeline_configs = configs["train_configs"]["data_loader"]["dataset"]
    train_pipeline_configs["phase"] = "train"
    validation_pipeline_configs = train_pipeline_configs.copy()
    validation_pipeline_configs["phase"] = "validation"

    train_pipeline = DaliFullPipeline(**train_pipeline_configs)
    train_pipeline.build()
    validation_pipeline = DaliFullPipeline(**validation_pipeline_configs)
    validation_pipeline.build()

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
    dice_metric = DiceMetric(n_class=3, brats=True)
    optmizer = OPTIMIZERS[train_configs.optimizer["name"]](
        model.parameters(), **train_configs.optimizer["parameters"]
    )

    scheduler = get_scheduler(
        scheduler_fn=train_configs.scheduler.scheduler_fn,
        optimizer=optmizer,
        t_total=train_configs.epochs * int(
            np.ceil(
                train_pipeline.nift_iterator.dataset_len \
                    / train_configs.batch_size
            )
        ),
        from_monai=train_configs.scheduler.from_monai,
        **train_configs.scheduler.scheduler_kwargs
    )

    train_loop(
        model, train_pipeline, validation_pipeline, optmizer, loss, dice_metric,
        scheduler, train_configs
    )

def save_prediction(image, path_to_save):
    image = torch.sigmoid(image.cpu())
    prediction_final = np.zeros((240, 240, 155), np.uint8)
    idx = image[0, 0] > 0.5
    prediction_final[idx] = 1
    idx = image[0, 1] > 0.5
    prediction_final[idx] = 2
    idx = image[0, 2] > 0.5
    prediction_final[idx] = 3

    niivol = nib.Nifti1Image(prediction_final, np.eye(4))
    niivol.header.get_xyzt_units()
    niivol.to_filename(path_to_save)


def evaluate(model, pipeline, phase):
    dice_calculator = DiceMetric(3, True)

    steps = int(
        np.ceil(pipeline.nift_iterator.dataset_len)
    )

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to("cuda")
        model.eval()
        to_csv = [["image_id", "dice_score"]]

        with trange(steps, desc="Validation Loop") as progress_bar:
            for batch_idx in progress_bar:
                volumetric_images, segmentation_masks = pipeline.run()
                volumetric_images = torch.Tensor(
                    volumetric_images.as_cpu().as_array()
                ).to("cuda")
                segmentation_masks = torch.Tensor(
                    segmentation_masks.as_cpu().as_array()
                ).to("cuda")
                predicted_segmentation = model(volumetric_images)

                dice = torch.mean(
                    dice_calculator.compute(
                        predicted_segmentation, segmentation_masks
                    )
                ).cpu()

                image_id = pipeline.nift_iterator.seen_data[batch_idx]
                to_csv.append([image_id, dice])

    with open(f"model_{phase}.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(to_csv)

def validate(configs: Dict):
    torch.cuda.set_device(1)
    train_pipeline_configs = configs["train_configs"]["data_loader"]["dataset"]
    train_pipeline_configs["phase"] = "train"
    train_pipeline_configs["batch_size"] = 1
    train_pipeline_configs["crop"] = False
    validation_pipeline_configs = train_pipeline_configs.copy()
    validation_pipeline_configs["phase"] = "validation"
    validation_pipeline_configs["batch_size"] = 1
    train_pipeline_configs["crop"] = False
    test_pipeline_configs = train_pipeline_configs.copy()
    test_pipeline_configs["phase"] = "test"
    test_pipeline_configs["batch_size"] = 1
    train_pipeline_configs["crop"] = False

    train_pipeline = DaliFullPipeline(**train_pipeline_configs)
    train_pipeline.build()
    validation_pipeline = DaliFullPipeline(**validation_pipeline_configs)
    validation_pipeline.build()
    test_pipeline = DaliFullPipeline(**train_pipeline_configs)
    test_pipeline.build()

    checkpoint_path = configs["train_configs"]["checkpoint"]

    model = get_model(configs)
    model.load_state_dict(torch.load(checkpoint_path))

    evaluate(model, train_pipeline, "train")
    evaluate(model, validation_pipeline, "validation")
    evaluate(model, test_pipeline, "test")


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
    elif args.mode == "validation":
        validate(configs)
