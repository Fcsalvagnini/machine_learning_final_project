# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from monai.metrics import compute_dice, do_metric_reduction
from monai.networks.utils import one_hot

class DiceMetric():
    def __init__(self, n_class, brats):
        self.n_class = n_class
        self.brats = brats

    def compute(self, p, y):
        if self.brats:
            p = (torch.sigmoid(p) > 0.5).int()
            y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
            y = torch.stack([y_wt, y_tc, y_et], dim=1)
        else:
            p, y = self.ohe(torch.argmax(p, dim=1)), self.ohe(y)

        y = y.squeeze()
        # Removes first dimension, when dealing of a batch size of 1
        p = p.squeeze()
        dice = self.compute_metric(p, y, compute_dice, 1, 0)

        return dice * 100

    def ohe(self, x):
        return one_hot(x, num_classes=self.n_class + 1, dim=1)

    def compute_metric(self, p, y, metric_fn, best_metric, worst_metric):
        """If batch has only one image, returns batch dimmension after squeeze,
        for correct metric calculation"""
        if p.dim() == 4:
            p = p.unsqueeze(0)
            y = y.unsqueeze(0)

        metric = metric_fn(p, y, include_background=self.brats)
        metric = torch.nan_to_num(metric, nan=worst_metric, posinf=worst_metric, neginf=worst_metric)
        metric = do_metric_reduction(metric, "mean_batch")[0]

        for i in range(self.n_class):
            if (y[:, i] != 1).all():
                metric[i - 1] += best_metric if (p[:, i] != 1).all() else worst_metric

        return metric
