
import itertools
import os

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from .pipeline import GenericPipeline

class TrainPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.oversampling = kwargs["oversampling"]
        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)