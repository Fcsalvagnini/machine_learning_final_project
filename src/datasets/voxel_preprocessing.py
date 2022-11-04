import json
import matplotlib.pyplot as plt
import monai as mn
import nibabel as nib 
import numpy as np
from send2trash import TrashPermissionError
import torch
import random

from typing import Dict, List
from typing import Optional, Callable, Tuple, Literal

from src.augmentation.augmentations import AugmentationPipeline
class BrainPreProcessing:

    def _nib_load_images(self, image_path: str, in_img: bool = True) -> np.ndarray:
        voxel_nii = nib.load(image_path)
        voxel_data = voxel_nii.get_data() #.dataobj
        voxel_np = np.asarray(voxel_data)
        if not in_img:
            voxel_np[voxel_np == 4] = 3

        return voxel_np
    
    @staticmethod
    def crop_foreground(image):
        return mn.transforms.CropForeground()(image)


    @staticmethod
    def normalize(image):
        return mn.transforms.NormalizeIntensity()(image)


    def to_tensor_transform(self, x, y):
        return AugmentationPipeline({
            "ToTensor": {
                "apply_on_label": True,
                "track_meta": False,
            }
        })(x, y)



    @staticmethod
    def random_spatial_crop(self, voxel_homo_size: int = 128) -> Callable:
        return mn.transforms.RandSpatialCrop(
            roi_size=[voxel_homo_size]*3,
            random_center=True,
            random_size=False
        )

    def _concatenate_voxel(self, *voxels: List[np.ndarray], axis: int=0) -> torch.Tensor:
        return torch.cat(*voxels, axis=axis)




    def prepare_nib_data(
        self,
        images_path: List[str],
        preprocess_fn: Callable,
        as_torch_tensor: bool = True,
        dtype: Literal["uint8", "int16"] = "uint8",
        in_img: bool = False
        ) -> torch.Tensor:
        
        voxels = list(
            map(lambda image_path: self._nib_load_images(image_path=image_path, in_img=in_img), images_path))

        if (preprocess_fn):
            voxels = list(map(lambda voxel: np.expand_dims(voxel, axis=0), voxels))
            if in_img:
                voxels = list(map(lambda voxel: BrainPreProcessing.normalize(voxel), voxels))        
            voxels = list(map(lambda voxel: preprocess_fn(voxel), voxels))
    
        tensor_dtype = getattr(torch, dtype)
        if as_torch_tensor:
            voxels = list(map(lambda voxel: voxel.as_tensor().type(tensor_dtype), voxels))
        if len(images_path) > 1:
            conc_voxel = self._concatenate_voxel(voxels)    
            return conc_voxel
        else:
            return voxels[0]