import json
import matplotlib.pyplot as plt
import monai as mn
import nibabel as nib 
import numpy as np
import torch
import random

from typing import Dict, List
from typing import Optional, Callable, Tuple

from src.utils.viewers.viewer_2d import multi_slice_viewer



def nib_load_images(image_path: str) -> np.ndarray:
    voxel_nii = nib.load(image_path)
    voxel_data = voxel_nii.get_data() #.dataobj
    voxel_np = np.asarray(voxel_data)
    return voxel_np

def prepare_nib_data(
    images_path: List[str],
    voxel_homo_size: int = 128
    ) -> np.ndarray:
    voxels = list(map(lambda img: nib_load_images(img), images_path)) #List[np.ndarray, np.ndarray]
    if (voxel_homo_size):
        voxels = list(map(lambda voxel: np.expand_dims(voxel, axis=0), voxels))
        crop = random_spatial_crop(voxel_homo_size=voxel_homo_size)

        voxels = list(map(lambda voxel: crop(voxel), voxels))
    
    if len(images_path) > 1:
        conc_voxel = concatenate_voxel(voxels)    
        return conc_voxel
    else:
        return voxels[0]


def random_spatial_crop(voxel_homo_size: int = 128) -> Callable:

    return mn.transforms.RandSpatialCrop(
        roi_size=[voxel_homo_size]*3,
        random_center=True,
        random_size=False
    )


def concatenate_voxel(*voxels: List[np.ndarray], axis: int=0) -> np.ndarray:
    return torch.cat(*voxels, axis=axis)


def define_brats_types_idxs(num: int = 2) -> List[int]: 
    indeces = []
    while len(indeces) < num:
        rnd = random.randint(0, 3)
        if not rnd in indeces:
            indeces.append(rnd)
    return indeces


if __name__ == "__main__":
    img_path = "/home/nakano/unicamp/master/MO444/project/machine_learning_final_project/datasets/BraTS2021_00253/BraTS2021_00253_flair.nii.gz"
    nib_load_images(image_path=img_path)
    
    