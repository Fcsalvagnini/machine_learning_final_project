
from multiprocessing.sharedctypes import Value
import nibabel as nib 
import os

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List
from typing import Callable, Optional, Sequence, Tuple, Union, Literal

from . import JsonHandler
from .voxel_preprocessing import BrainPreProcess

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
        json_path: Union[str, Path],
        data_path: Union[str, Path],
        num_concat: Literal[2, 4] = 2,
        transforms: Optional[Callable] = None,
        voxel_homo_size: int = 128
        ) -> None:
        self._json_path = json_path
        self._data_path = data_path
        self._num_concat = num_concat 
        self._transforms = transforms
        self._voxel_homo_size = voxel_homo_size

        self.brats_types = ["flair", "t1", "t1ce", "t2"]
        self.gt_brats_types = ["seg"]
        
        self._brats_ids = self._get_phase_ids() 
        self._brats_types_concat = self._get_brats_types_concat()
        
        self.brain_preprocess = BrainPreProcess()

    def _get_phase_ids(self) -> List:
        return JsonHandler.parse_json_to_dict(
            json_path=self._json_path)["ids"]
    
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
        
        fn_random_spatial_crop = self.brain_preprocess.random_spatial_crop(self._voxel_homo_size)
        
        x_brats = self.brain_preprocess.prepare_nib_data(
            images_path=x_brats_files, 
            preprocess_fn=fn_random_spatial_crop
        )
        
        y_brats = self.brain_preprocess.prepare_nib_data(
            images_path=y_brats_file,
            preprocess_fn=fn_random_spatial_crop
        )
         
        if self._transforms:
            raise NotImplementedError("Augmentation method not implemented yet.")
        
        return x_brats, y_brats 


    def __len__(self) -> int:
        return len(self._brats_ids)


if __name__ == "__main__":
    dataset_kwargs = {
        "json_path": "src/data/descriptors/test.json",
        "data_path": "datasets",
        "num_concat": 4,
        "transforms": None,
        "voxel_homo_size":  128
    }
    
    dataset = BrainDataset(**dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )

    interator = iter(dataloader)
    x, y = next(interator)
    print("x shape: ", x.shape)
    print("y shape: ", y.shape)
    
