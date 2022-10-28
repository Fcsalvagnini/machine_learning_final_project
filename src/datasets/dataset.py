
from multiprocessing.sharedctypes import Value
import nibabel as nib 
import os

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List
from typing import Callable, Optional, Sequence, Tuple, Union, Literal

from . import JsonHandler
from .voxel_preprocessing import BrainPreProcess
from src.augmentation.augmentations import AugmentationPipeline
"""
TODO:
    [ ] - augmentations
        [ ] - methods?
        [ ] - must the mask be passed as well?
    [X] - ToTensor -> output from preprocess is already a torch.Tensor
"""   

class BrainDataset(Dataset):
    def __init__(
        self, 
        cfg: Dict,
        phase: Literal["train", "valid", "test"],
        num_concat: Literal[2, 4] = 2,
        #data_path: Union[str, Path],
        #transforms: Optional[Callable] = None,
        #voxel_homog_size: int = 128
        ) -> None:
        print(cfg)
        self._phase = phase
        self._num_concat = num_concat 
        self._data_path = cfg.data_path
        self._transforms = cfg.transforms
        self._voxel_homog_size = cfg.voxel_homog_size

        self.brats_types = ["flair", "t1", "t1ce", "t2"]
        self.gt_brats_types = ["seg"]
        
        print(self._transforms.augmentations)
        

        self._transforms = AugmentationPipeline(cfg.transforms.augmentations)
        self._brats_ids = self._get_phase_ids() 
        self._brats_types_concat = self._get_brats_types_concat()
        
        self.brain_preprocess = BrainPreProcess()

    def _get_phase_ids(self) -> List:
        return JsonHandler.parse_json_to_dict(
            phase=self._phase)["ids"]
    
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
        
        fn_random_spatial_crop = self.brain_preprocess.random_spatial_crop(self._voxel_homog_size)
        
        x_brats = self.brain_preprocess.prepare_nib_data(
            images_path=x_brats_files, 
            preprocess_fn=fn_random_spatial_crop,
            as_torch_tensor=True
        )
        
        y_brats = self.brain_preprocess.prepare_nib_data(
            images_path=y_brats_file,
            preprocess_fn=fn_random_spatial_crop,
            as_torch_tensor=True
        )
         
        if self._transforms:
            #raise NotImplementedError("Augmentation method not implemented yet.")
            x_brats, y_brats = self._transforms(x_brats, y_brats)

        return x_brats, y_brats 


    def __len__(self) -> int:
        return len(self._brats_ids)


if __name__ == "__main__":
    dataset_kwargs = {
        "phase": "test",
        "data_path": "database",
        "num_concat": 4,
        "transforms": None,
        "voxel_homog_size":  128
    }
    from src.utils.yaml.yaml_handler import YamlHandler
    from src.utils.configurator import DatasetConfigs
    cfg = YamlHandler.parse_yaml_to_dict("nn_unet_nvidia.yaml")
    #print(cfg)
    ds_cfg = DatasetConfigs(cfg["train_configs"]["data_loader"]["dataset"])
    print(ds_cfg.__dict__)

    dataset = BrainDataset(ds_cfg, phase="test", num_concat=2)
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=False
    )

    interator = iter(dataloader)
    x, y = next(interator)
    
    