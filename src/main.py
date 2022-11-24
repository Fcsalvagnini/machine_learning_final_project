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

#from src.utils.wandb.runner import WandbArtifactRunner
#from src.utils.wandb.logger import WandbLogger

from torch.utils.data import DataLoader

def run_train_epoch(model, optimizer, loss, dice_metric, dataloader, monitoring_metrics,
                epoch):
    model.train()
    model.to("cuda")
    running_loss = 0
    running_dice = 0

    with trange(len(dataloader), desc="Train Loop") as progress_bar:
        for batch_idx, batch in zip(progress_bar, dataloader):
            volumetric_image, segmentation_mask = [
                data.to("cuda") for data in batch
            ]
            optimizer.zero_grad()
            predicted_segmentation = model(volumetric_image)
            
            batch_loss = loss.forward(predicted_segmentation, segmentation_mask)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.cpu()
            running_dice += torch.mean(dice_metric.compute(predicted_segmentation, segmentation_mask)).cpu()

            progress_bar.set_postfix(
                desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f} | Dice: {running_dice / (batch_idx + 1):.3f}"
            )

    epoch_loss = (running_loss / len(dataloader)).detach().numpy()
    monitoring_metrics["loss"]["train"].append(epoch_loss)

    return epoch_loss

def run_validation_epcoch(model, optimizer, loss, dice_metric, dataloader, 
                monitoring_metrics, epoch, configurations, 
                save_best_model):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to("cuda")
        model.eval()
        running_loss = 0
        running_dice = 0

        with trange(len(dataloader), desc="Validation Loop") as progress_bar:
            for batch_idx, batch in zip(progress_bar, dataloader):
                volumetric_image, segmentation_mask = [
                    data.to("cuda") for data in batch[0]
                ]
                predicted_segmentation = model(volumetric_image)
                batch_loss = loss.forward(predicted_segmentation, segmentation_mask)

                running_loss += batch_loss.cpu()
                running_dice += torch.mean(dice_metric.compute(predicted_segmentation, segmentation_mask)).cpu()

                progress_bar.set_postfix(
                    desc=f"[Epoch {epoch}] Loss: {running_loss / (batch_idx + 1):.3f} | Dice: {running_dice / (batch_idx + 1):.3f}"
                )

    epoch_loss = (running_loss / len(dataloader)).detach().numpy()
    monitoring_metrics["loss"]["validation"].append(epoch_loss)

    save_best_model(epoch_loss, epoch, model, configurations)

    return epoch_loss

def train_loop(model, train_dataloader, validation_dataloader, optmizer, loss,
            dice_metric, scheduler, train_configs):
    os.makedirs(train_configs.checkpoints_path, exist_ok=True)
    monitoring_metrics = {
        "loss": {"train": [], "validation": []}
    }
    save_best_model = SaveBestModel()

    for epoch in range(1, train_configs.epochs + 1):
        train_loss = run_train_epoch(
            model, optmizer, loss, dice_metric, train_dataloader, monitoring_metrics,
            epoch
        )
        valid_loss = run_validation_epcoch(
            model, optmizer, loss, dice_metric, validation_dataloader, monitoring_metrics,
            epoch, train_configs, save_best_model
        )
        #scheduler.step(monitoring_metrics["loss"]["validation"][-1])

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
    dice_metric = DiceMetric(n_class=3, brats=True)
    optmizer = OPTIMIZERS[train_configs.optimizer["name"]](
        model.parameters(), **train_configs.optimizer["parameters"]
    )
    #scheduler = get_scheduler(
    #    train_configs.scheduler.scheduler_fn,
    #    optimizer=optmizer, from_monai=train_configs.scheduler.from_monai,
    #    **train_configs.scheduler.scheduler_kwargs
    #)
    scheduler=None

    train_loop(
        model, train_dataloader, validation_dataloader, optmizer, loss, dice_metric,
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


def validate(configs: Dict):
    torch.cuda.set_device(1)
    train_configs = TrainConfigs(configs["train_configs"])
    validation_configs = ValidationConfigs(configs["validation_configs"])
    validation_dataset_configs = DatasetConfigs(configs["validation_configs"]["data_loader"]["dataset"])
    validation_test_dataset = BrainDataset(
        cfg=validation_dataset_configs,
        phase="validation_test",
        num_concat=2
    )
    validation_test_dataloader = DataLoader(
        validation_test_dataset, batch_size=1
    )

    model = get_model(configs)
    model.load_state_dict(torch.load(validation_configs.checkpoint_path))
    dice_calculator = DiceMetric(3, True)

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to("cuda")
        model.eval()
        to_csv = [["image_path", "dice_score"]]

        for image in tqdm(validation_test_dataloader):
            (x, y), path = image
            prediction = sliding_window_inference(x.to("cuda"), [128, 128, 128], 1, model)
            dice_pred = torch.mean(dice_calculator.compute(prediction, y.to("cuda")))
            prediction_save_path = f"{path[0]}_pred_seg.nii.gz"
            folder_path = os.path.join(*prediction_save_path.split("/")[:-1])
            save_prediction(prediction, prediction_save_path)

            to_csv.append([folder_path, dice_pred.cpu().detach().numpy()])

    with open("best_model_without_aug.csv", "w") as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerows(to_csv)


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
