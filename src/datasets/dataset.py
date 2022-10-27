
import nibabel as nib 
import os

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List
from typing import Callable, Optional, Sequence, Tuple, Union

from . import JsonHandler

from utils import define_brats_types_idxs, prepare_nib_data

"""
TODO:
    [ ] - augmentations
    [ ] - ToTensor
    [ ] - Test with real data
"""

class BrainDataset(Dataset):
    def __init__(
        self, 
        data_file: Union[str, Path],
        data_path: Union[str, Path],
        num_concat: int = 2,
        transforms: Optional[Callable] = None,
        crop_size: Tuple[int, int, int] = (128, 128, 128)
        ) -> None:
        self.data_file = data_file
        self.data_path = data_path
        self.num_concat = num_concat 
        self.transforms = transforms
        self.crop_size = crop_size

        self.brats_types = ["flair", "t1", "t1ce", "t2"]
        self.gt_brats_types = ["seg"]
        
        self.brats_ids = self._get_phase_ids() 
        

    def _get_phase_ids(self) -> List:
        return JsonHandler.parse_json_to_dict(
            json_path=os.path.join(self.data_path, self.data_path)
        )
    
    def __getitem__(self, idx) -> Sequence:
        brats_id = self.brats_ids[idx]
        x_brats_types_chosen = list(map(lambda idx: self.brats_types[idx], define_brats_types_idxs(self.num_concat)))
        x_brats_files = list(map(lambda brats_type: os.path.join(
            self.data_path, f"{brats_id}/{brats_id}_{brats_type}.nii.gz"), 
            x_brats_types_chosen))
        x_brats = prepare_nib_data(images_path=x_brats_files, voxel_homo_size=128)
        
        y_brats_file = os.path.join(self.data_path, f"{brats_id}/{brats_id}_seg.nii.gz")
        y_brats = prepare_nib_data(images_path=y_brats_file, voxel_homo_size=128)
         
        if self.transforms:
            raise NotImplementedError("Augmentation method not implemented yet.")

        #return 


    def __len__(self) -> int:
        return len(self.brats_id)
        