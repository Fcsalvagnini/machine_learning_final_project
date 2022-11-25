import os
import random
from typing import Tuple, Literal

import torch
import torchio as tio
import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

from monai.transforms import CropForeground
from monai.transforms import SpatialPad

from src.utils.json.json_handler import parse_json_to_dict
from src.dali_dataloader.utils import probabilistic_augmentation

class NiftiIterator(object):
    def __init__(self, image_paths:str, label_paths:str, crop:bool=True) -> None:
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.crop = crop
        self.dataset_len = len(self.image_paths)
        # Attribute to track image ids for each data pair
        self.seen_data = []

        if self.crop:
            # Objects to tackle image pre-processing
            self.cropper = CropForeground(
                select_fn=lambda x: x != 0, margin=0, return_coords=True
            )
            self.padder = SpatialPad(spatial_size=(128, 128, 128))

    def __iter__(self) -> object:
        self.idx = 0

        return self

    def __next__(self) -> Tuple:
        if self.idx == 0:
            self._shuffle(self.image_paths, self.label_paths)

        image_modalities_paths = self.image_paths[self.idx]
        label_path = self.label_paths[self.idx]
        self.seen_data.append(label_path.split("/")[-2])

        image_modalities_data = []
        for path in image_modalities_paths:
            image_data = self._read_nifti_image(path)
            image_data = self._normalize(image_data)
            image_modalities_data.append(image_data)

        image = torch.cat(
            image_modalities_data, axis=0
        ).to(torch.float)
        label = self._read_nifti_image(label_path).to(torch.uint8)

        if self.crop:
            bbox_start, bbox_end = self.cropper.compute_bounding_box(image)
            image = image[
                0:len(image_modalities_paths),
                bbox_start[0]:bbox_end[0],
                bbox_start[1]:bbox_end[1],
                bbox_start[2]:bbox_end[2]
            ].contiguous()
            label = label[
                :1,
                bbox_start[0]:bbox_end[0],
                bbox_start[1]:bbox_end[1],
                bbox_start[2]:bbox_end[2]
            ].contiguous()

            # Verify if croped image is smaller than 128, 128, 128 and pad it
            image = self.padder(image).as_tensor()
            label = self.padder(label).as_tensor()

        """Iterates to the next sample, or if reached the end, reshufle
        paths and start again.
        """
        self.idx = (self.idx + 1) % self.dataset_len

        return image, label

    def _normalize(self, image:torch.Tensor) -> torch.Tensor:
        image = image.to(torch.float)
        non_zero_voxels = image[image != 0]
        mean = torch.mean(non_zero_voxels)
        std = torch.std(non_zero_voxels)

        normalized_image = image
        normalized_image[image != 0] = (non_zero_voxels - mean) / std

        return normalized_image

    def _shuffle(self, image_paths:str, label_paths:str) -> None:
        temp_list = list(zip(image_paths, label_paths))
        random.shuffle(temp_list)
        image_paths, label_paths = zip(*temp_list)

        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)

    def _read_nifti_image(self, image_path:str) -> torch.Tensor:
        image_data = tio.ScalarImage(image_path)[tio.DATA]

        return image_data

    def reset(self) -> None:
        self.seen_data = self.seen_data[self.dataset_len:]

class GenericPipeline(Pipeline):
    def __init__(self, data_path:str, data_descriptors_path:str,
            phase: Literal["train", "validation", "test"], n_modalities:int,
            batch_size:int, num_threads:int, device_id:int,
            patch_size:Tuple, crop:bool, evaluate:bool
        ):
        super().__init__(batch_size, num_threads, device_id)
        self.data_path = data_path
        self.data_descriptors_path = data_descriptors_path
        self.phase = phase
        self.n_modalities = n_modalities
        self.patch_size = patch_size
        self.evaluate = evaluate

        image_paths, label_paths = self._get_image_paths(
            data_path=self.data_path,
            data_descriptors_path=self.data_descriptors_path,
            phase=self.phase,
            n_modalities=self.n_modalities
        )

        self.nift_iterator = NiftiIterator(
            image_paths=image_paths,
            label_paths=label_paths,
            crop=crop
        )

    def _get_image_paths(self, data_path:str, data_descriptors_path:str,
            phase:Literal["train", "validation", "test"],
            n_modalities:int
        ) -> Tuple:
        subject_paths = parse_json_to_dict(
            data_descriptors_path=data_descriptors_path,
            phase=phase
        )["ids"]

        if n_modalities == 2:
            modalities = ["flair", "t1ce"]
        elif n_modalities == 4:
            modalities = ["flair", "t1", "t1ce", "t2"]
        else:
            raise ValueError(
                f"Number of Modalities must be 2 or 4. Received {n_modalities}"
            )

        image_paths = list(
            map(lambda subject_path: [
                f"{data_path}/{subject_path}/{subject_path}_{modality}.nii.gz" \
                    for modality in modalities
            ], subject_paths)
        )
        label_paths = list(
            map(lambda subject_path:
                f"{data_path}/{subject_path}/{subject_path}_seg.nii.gz"
            , subject_paths)
        )

        return image_paths, label_paths

    def _crop(self, data:torch.Tensor) -> torch.Tensor:
        return fn.crop(data, crop=self.patch_size, out_of_bounds_policy="pad")

    def _crop_fn(self, image:torch.Tensor,
            label:torch.Tensor) -> Tuple:
        image, label = self.crop(image), self.crop(label)

        return image, label

class DaliFullPipeline(GenericPipeline):
    def __init__(self, data_path:str, data_descriptors_path:str,
            phase: Literal["train", "validation", "test"], n_modalities:int,
            batch_size:int, num_threads:int, device_id:int,
            patch_size:Tuple, crop:bool, evaluate:bool
        ) -> None:
        super().__init__(
            data_path=data_path, data_descriptors_path=data_descriptors_path,
            phase=phase, n_modalities=n_modalities, batch_size=batch_size,
            num_threads=num_threads, device_id=device_id,
            patch_size=patch_size, crop=crop, evaluate=evaluate
        )
        self.crop_shape = types.Constant(
            np.array(self.patch_size), dtype=types.INT64
        )
        self.crop_shape_float = types.Constant(
            np.array(self.patch_size), dtype=types.FLOAT
        )

    def _biased_crop_fn(self, image:torch.Tensor, label:torch.Tensor
        ) -> Tuple:
        # With probability of 0.4 the patch selected via random biased crop is
        # going to hold foreground voxels.
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            label, background=0, format="start_end", cache_objects=False,
            foreground_prob=0.4
        )
        # Generates a Random Crop Window which coints the roi defined by
        # random_object_bbox.
        anchor = fn.roi_random_crop(
            label, roi_start=roi_start, roi_end=roi_end,
            crop_shape=[1, *self.patch_size]
        )
        # Drop channels from anchor
        anchor = fn.slice(anchor, 1, 3, axes=[0])
        image, label = fn.slice(
            [image, label], anchor, self.crop_shape, axis_names="DHW",
            out_of_bounds_policy="pad"
        )

        return image.gpu(), label.gpu()

    def _resize(self, data:torch.Tensor,
            interpolation_type:types.DALIInterpType.INTERP_CUBIC
        ) -> Tuple:
        return fn.resize(data, interp_type=interpolation_type, size=self.crop_shape_float)

    def _zoom_fn(self, image:torch.Tensor, label:torch.Tensor
        ) -> Tuple:
        scale = probabilistic_augmentation(0.15, fn.random.uniform(range=(1.0, 1.4)), 1.0)
        c, h, w = [scale * x for x in self.patch_size]

        image = fn.crop(image, crop_h=h, crop_w=w, crop_d=c, out_of_bounds_policy="pad")
        label = fn.crop(label, crop_h=h, crop_w=w, crop_d=c, out_of_bounds_policy="pad")
        image = self._resize(image, types.DALIInterpType.INTERP_CUBIC)
        label = self._resize(label, types.DALIInterpType.INTERP_NN)

        return image, label

    def _flips_fn(self, image:torch.Tensor, label:torch.Tensor
        ) -> Tuple:
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
            "depthwise": fn.random.coin_flip(probability=0.5)
        }

        return fn.flip(image, **kwargs), fn.flip(label, **kwargs)

    def _noise_fn(self, image:torch.Tensor) -> torch.Tensor:
        image_noised = image + fn.random.normal(image, stddev=fn.random.uniform(range=(0.0, 0.33)))

        return probabilistic_augmentation(0.15, image_noised, image)

    def _blur_fn(self, image:torch.Tensor) -> torch.Tensor:
        image_blurred = fn.gaussian_blur(image, sigma=fn.random.uniform(range=(0.5, 1.5)))

        return probabilistic_augmentation(0.15, image_blurred, image)

    def _brightness_fn(self, image:torch.Tensor) -> torch.Tensor:
        brightness_scale = probabilistic_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        image = image * brightness_scale

        return image

    def _contrast_fn(self, image:torch.Tensor) -> torch.Tensor:
        scale = probabilistic_augmentation(0.15, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        image = math.clamp(image * scale, fn.reductions.min(image), fn.reductions.max(image))

        return image

    def define_graph(self) -> Tuple:
        image, label = fn.external_source(
            source=self.nift_iterator, num_outputs=2,
            dtype=[types.FLOAT, types.UINT8], batch=False
        )
        image = fn.reshape(image, layout="CDHW")
        label = fn.reshape(label, layout="CDHW")
        image, label = self._biased_crop_fn(image, label)
        if self.phase == "train" and not self.evaluate:
            image, label = self._zoom_fn(image, label)
            image, label = self._flips_fn(image, label)
            image = self._noise_fn(image)
            image = self._blur_fn(image)
            image = self._brightness_fn(image)
            image = self._contrast_fn(image)

        return (image, label)