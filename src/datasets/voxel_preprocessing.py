import json
import matplotlib.pyplot as plt
import monai as mn
import nibabel as nib 
import numpy as np
import torch
import random

from typing import Dict, List
from typing import Optional, Callable, Tuple

class BrainPreProcess:
    def __init__(
        self,
        #voxel_homo_size: int = 128
        ) -> None:
        #self._voxel_homo_size = voxel_homo_size
        #self._random_spatial_crop = self.random_spatial_crop()
        pass

    def _nib_load_images(self, image_path: str) -> np.ndarray:
        voxel_nii = nib.load(image_path)
        voxel_data = voxel_nii.get_data() #.dataobj
        voxel_np = np.asarray(voxel_data)
        return voxel_np
    
    @staticmethod
    def random_spatial_crop(self, voxel_homo_size: int = 128) -> Callable:
        return mn.transforms.RandSpatialCrop(
            roi_size=[voxel_homo_size]*3,
            random_center=True,
            random_size=False
        )

    def _concatenate_voxel(self, *voxels: List[np.ndarray], axis: int=0) -> np.ndarray:
        return torch.cat(*voxels, axis=axis)

    def prepare_nib_data(
        self,
        images_path: List[str],
        preprocess_fn: Callable
        ) -> np.ndarray:
        voxels = list(map(lambda img: self._nib_load_images(img), images_path))
        if (preprocess_fn):
            voxels = list(map(lambda voxel: np.expand_dims(voxel, axis=0), voxels))
            voxels = list(map(lambda voxel: preprocess_fn(voxel), voxels))

        if len(images_path) > 1:
            conc_voxel = self._concatenate_voxel(voxels)    
            return conc_voxel
        else:
            return voxels[0]