
from multiprocessing.sharedctypes import Value
import nibabel as nib 
import os

import numpy as np

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List
from typing import Callable, Optional, Sequence, Tuple, Union, Literal

from monai import transforms as mtransforms
from . import parse_json_to_dict
from .preprocessing import BrainPreProcessing

from src.augmentation.augmentations import AugmentationPipeline
from src.utils.configurator import DatasetConfigs
from src.utils.viewers.viewer_2d import multi_slice_viewer

class BrainDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfigs,
        phase: Literal["train", "validation", "test"],
        num_concat: Literal[2, 4] = 2,
        #data_path: Union[str, Path],
        #transforms: Optional[Callable] = None,
        #voxel_homog_size: int = 128
        ) -> None:
        self._phase = phase
        self._num_concat = num_concat
        self._data_path = cfg.data_path
        self._data_descriptors_path = cfg.data_descriptors_path
        self._transforms = cfg.transforms
        self._voxel_homog_size = cfg.voxel_homog_size

        self.brats_types = ["flair", "t1", "t1ce", "t2"]
        self.gt_brats_types = ["seg"]

        if self._phase == "train":
            self._transforms = AugmentationPipeline(cfg.transforms.augmentations)
        else:
            self._transforms = None

        self._brats_ids = self._get_phase_ids()
        self._brats_types_concat = self._get_brats_types_concat()

        self._brain_preprocess = BrainPreProcessing()
        

    def _get_phase_ids(self) -> List:
        _phase = self._phase
        if _phase == "validation_test":
            _phase = "validation"
        return parse_json_to_dict(
            self._data_descriptors_path, phase=_phase
        )["ids"]

    def _get_brats_types_concat(self):
        if self._num_concat == 2:
            return ["flair", "t1ce"]
        elif self._num_concat == 4:
            return ["flair", "t1", "t1ce", "t2"]
        else:
            raise ValueError(f"num_concat variable must be 2 or 4. Value passed is: {self._num_concat}")



    def __getitem__(self, idx) -> Sequence:

        brats_id = self._brats_ids[idx]

        x_brats_files = list(map(lambda brats_type: os.path.join(
            self._data_path, f"{brats_id}/{brats_id}_{brats_type}.nii.gz"),
            self._brats_types_concat))
        y_brats_file = list(map(lambda gt_brats_type:
            os.path.join(self._data_path, f"{brats_id}/{brats_id}_{gt_brats_type}.nii.gz"), self.gt_brats_types))

        if self._phase == "test" or self._phase == "validation_test":
            preprocess_fn = None
        else:
            preprocess_fn = BrainPreProcessing.random_spatial_crop(self._voxel_homog_size)
            
        x_brats = self._brain_preprocess.load_data(
            images_path=x_brats_files,
            #preprocess_fn=preprocess_fn,
            as_torch_tensor=True,
            in_img=True,
            dtype="float32"
        )
        
        y_brats = self._brain_preprocess.load_data(
            images_path=y_brats_file,
            #preprocess_fn=preprocess_fn,
            as_torch_tensor=True,
            dtype="uint8"
        )
        
        if (preprocess_fn):
            brats_conc = torch.cat([x_brats, y_brats], axis=0)        
            brats_conc = preprocess_fn(brats_conc)
            x_brats, y_brats = brats_conc[:2], brats_conc[2:]

        if self._transforms and self._phase == "train":    
            x_brats, y_brats = self._transforms(x_brats, y_brats)
            x_brats, y_brats = x_brats.type(torch.float32), y_brats.type(torch.int8)
            to_return = (x_brats, y_brats)
        else:
            x_brats, y_brats = self._brain_preprocess.to_tensor_transform(x_brats, y_brats) 
            x_brats, y_brats = x_brats.type(torch.float32), y_brats.type(torch.int8)
            to_return = (x_brats, y_brats), os.path.join(self._data_path, f"{brats_id}/{brats_id}")

        # x_brats_np, y_brats_np = x_brats.numpy(), y_brats.numpy()
        # xx = list(map(lambda x, name: save(x, name), x_brats, x_brats_files))
        # yy = list(map(lambda x, name: save(x, name), y_brats, y_brats_file))
        # xx0 = np.load(xx[0])
        # xx1 = np.load(xx[1])
        # yy0 = np.load(yy[0])
            
        # multi_slice_viewer(xx0, xx[0])
        # multi_slice_viewer(xx1, xx[1])
        # multi_slice_viewer(yy0, yy[0])
        
        return to_return

    def __len__(self) -> int:
        return len(self._brats_ids)

def save(image, name):
    if not (type) is np.ndarray:
        image = image.numpy()
    img_name = "result_images/" + os.path.join(name.replace(".nii.gz", ".npy").split("/")[-1])
    np.save(img_name, image, allow_pickle=False)
    return img_name

if __name__ == "__main__":
    dataset_kwargs = {
        "phase": "test",
        "data_path": "database",
        "num_concat": 2,
        "transforms": None,
        "voxel_homog_size":  128
    }
    from src.utils.yaml.yaml_handler import YamlHandler
    from src.utils.configurator import DatasetConfigs
    cfg = YamlHandler.parse_yaml_to_dict("nn_unet_nvidia.yaml")
    #print(cfg)
    ds_cfg = DatasetConfigs(cfg["train_configs"]["data_loader"]["dataset"])

    train_dataset = BrainDataset(ds_cfg, phase="train", num_concat=2)
    valid_dataset = BrainDataset(ds_cfg, phase="test", num_concat=2)

    batch_size = 1
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False
    )
    train_interator = iter(train_dataloader)
    #valid_interator = iter(valid_dataloader)
    for i in range(batch_size):
        
        #x_valid, y_valid = next(valid_interator)
        x_train, y_train = next(train_interator)

        print(f"x_max train: {torch.amax(x_train)}")
        #print(f"x_max test: {torch.amax(x_valid)}")
        print(f"y_max train: {torch.amax(y_train)}")
        #print(f"y_max test: {torch.amax(y_valid)}")
        print(f"x train type: {type(x_train)})")
        print(f"y train type: {type(y_train)})")
        #print(f"x test type: {type(x_valid)})")
        print(f"y test type: {type(y_train)})")
        print(f"x train type: {x_train.dtype})")
        print(f"y train type: {y_train.dtype})")
        #print(f"x test type: {x_valid.dtype})")
        print(f"y test type: {y_train.dtype})")
        print(f"x shape: {x_train.shape})")
        print(f"y shape: {y_train.shape})")
