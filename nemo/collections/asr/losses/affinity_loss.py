# ! /usr/bin/python
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.core.classes import Loss, Typing
from nemo.core.neural_types import LabelsType, LengthsType, LossType, NeuralType, ProbsType
from nemo.collections.asr.parts.utils.offline_clustering import cos_similarity_batch

class AffinityLoss(Loss, Typing):
    """
    Computes Binary Cross Entropy (BCE) loss. The BCELoss class expects output from Sigmoid function.
    """

    @property
    def input_types(self):
        """Input types definitions for AnguarLoss.
        """
        return {
            "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
            'labels': NeuralType(('B', 'T', 'C'), LabelsType()),
            "signal_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """
        Output types definitions for binary cross entropy loss. Weights for labels can be set using weight variables.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction='sum', gamma=0.0, negative_margin=0.5, positive_margin=0.05):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma # weights for loss values of [different, same] speaker segments
        self.negative_margin = negative_margin
        self.positive_margin = positive_margin

    def forward(self, batch_affinity_mat, targets):
        """
        Calculate binary cross entropy loss based on probs, labels and signal_lengths variables.

        Args:
            probs (torch.tensor)
                Predicted probability value which ranges from 0 to 1. Sigmoid output is expected.
            labels (torch.tensor)
                Groundtruth label for the predicted samples.
            signal_lengths (torch.tensor):
                The actual length of the sequence without zero-padding.

        Returns:
            loss (NeuralType)
                Binary cross entropy loss value.
        """
        gt_affinity = cos_similarity_batch(targets, targets)
        gt_affinity_margin = gt_affinity.clamp_(self.negative_margin, 1.0 - self.positive_margin)
        batch_affinity_mat_margin = batch_affinity_mat.clamp_(self.negative_margin, 1.0 - self.positive_margin)
        elem_affinity_loss = torch.abs(gt_affinity_margin - batch_affinity_mat_margin)
        positive_samples = gt_affinity * elem_affinity_loss
        negative_samples = (1-gt_affinity) * elem_affinity_loss
        affinity_loss = (1-self.gamma) * positive_samples.sum() + self.gamma * negative_samples.sum()
        return affinity_loss