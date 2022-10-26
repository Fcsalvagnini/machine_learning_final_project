import itertools
import os

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator



class GenericPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id)
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.device = device_id
        self.patch_size = kwargs["patch_size"]
        self.load_to_gpu = kwargs["load_to_gpu"]
        self.input_x = self.get_reader(kwargs["imgs"])
        self.input_y = self.get_reader(kwargs["lbls"]) if kwargs["lbls"] is not None else None

    def get_reader(self, data):
        return ops.readers.Numpy(
            files=data,
            device="cpu",
            read_ahead=True,
            dont_use_mmap=True,
            pad_last_batch=True,
            shard_id=self.device,
            seed=self.kwargs["seed"],
            num_shards=self.kwargs["gpus"],
            shuffle_after_epoch=self.kwargs["shuffle"],
        )

    def load_data(self):
        img = self.input_x(name="ReaderX")
        if self.load_to_gpu:
            img = img.gpu()
        img = fn.reshape(img, layout="CDHW")
        if self.input_y is not None:
            lbl = self.input_y(name="ReaderY")
            if self.load_to_gpu:
                lbl = lbl.gpu()
            lbl = fn.reshape(lbl, layout="CDHW")
            return img, lbl
        return img

    def crop(self, data):
        return fn.crop(data, crop=self.patch_size, out_of_bounds_policy="pad")

    def crop_fn(self, img, lbl):
        img, lbl = self.crop(img), self.crop(lbl)
        return img, lbl

    def transpose_fn(self, img, lbl):
        img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
        return img, lbl