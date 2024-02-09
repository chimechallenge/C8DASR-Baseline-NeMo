# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import sox
import os
import pickle as pkl
import tempfile
from collections import OrderedDict
from pathlib import Path
from statistics import mode
from typing import Any, Dict, List, Optional, Tuple, Union
from operator import attrgetter

import numpy as np
import time
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import OmegaConf

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations

from nemo.collections.asr.data.audio_to_msdd_label import AudioToSpeechMSDDInferDataset, AudioToSpeechMSDDTrainDataset, get_ms_seg_timestamps
from nemo.collections.asr.models.multi_classification_models import EncDecMultiClassificationModel
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.models.clustering_diarizer import (
    _MODEL_CONFIG_YAML,
    _SPEAKER_MODEL,
    _VAD_MODEL,
    get_available_model_names,
)


from nemo.collections.asr.models.configs.diarizer_config import NeuralDiarizerInferenceConfig
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.offline_clustering import (
    get_argmin_mat_large,
    cos_similarity_batch,
    SpeakerClustering,
)

from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_top_k_for_each_row,
    get_uniq_id_list_from_manifest,
    get_uniqname_from_filepath,
    parse_scale_configs,
    labels_to_pyannote_object,
    make_rttm_with_overlap,
    rttm_to_labels,
    get_selected_channel_embs,
)
from nemo.collections.asr.parts.utils.manifest_utils import (
read_manifest,
write_manifest,
)
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType, LabelsType, LossType
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['EncDecDiarLabelModel', 'ClusterEmbedding', 'NeuralDiarizer']

from nemo.core.classes import Loss, Typing, typecheck

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    pp_string = str(round(pp / float(10**6), 2)) + "M"
    print(f"size: {pp_string}")
    return pp

def check_multiscale_data(method):
    def wrapper(self, *args, **kwargs):
        for var_str in ['ms_seg_timestamps', 'ms_seg_counts', 'scale_mapping']:
            if torch.std(kwargs[var_str].float(), dim=0).sum() != 0:
                raise ValueError(f"Multi-scale variables should have the same values for all samples in a batch.")
            check_batch_unity(kwargs[var_str])
        return method(self, *args, **kwargs)
    return wrapper


def check_batch_unity(batch_tensor):
    if torch.std(batch_tensor.float(), dim=0).sum() != 0:
        raise ValueError(f"Multi-scale variables should have the same values for all samples in a batch.")

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
        # affinity_loss = (1-self.gamma) * positive_samples.sum() + self.gamma * negative_samples.sum()
        affinity_loss = (1-self.gamma) * positive_samples.sum() 
        return affinity_loss


def get_scale_dist_mat_t(source_scale_idx, ms_seg_timestamps, msdd_scale_n, deci=1):
    """
    Get distance matrix between anchors of the source scale and base scale.

    Args:
        source_scale_idx (int): Source scale index
        ms_seg_timestamps (Tensor): Multi-scale segment timestamps

    Returns:
        abs_dist_mat (Tensor): Distance matrix between anchors of the source scale and base scale
    """
    source_scale_anchor_zeros = torch.mean(ms_seg_timestamps[source_scale_idx, :, :], dim=1) / deci
    base_scale_anchor_zeros = torch.mean(ms_seg_timestamps[msdd_scale_n-1, :, :], dim=1) / deci
    # Get only non-zero timestamps (= Remove zero-padding)
    source_scale_anchor = source_scale_anchor_zeros[source_scale_anchor_zeros.nonzero()].t()
    base_scale_anchor = base_scale_anchor_zeros[base_scale_anchor_zeros.nonzero()].t()
    # Calculate absolute distance matrix
    curr_mat = torch.tile(source_scale_anchor, (base_scale_anchor.shape[0], 1))
    base_mat = torch.tile(base_scale_anchor, (source_scale_anchor.shape[0], 1)).t()
    abs_dist_mat = torch.abs(curr_mat - base_mat)
    return abs_dist_mat

def get_padded_timestamps_t(ms_ts_dict, base_scale):
    """
    Get zero-padded timestamps of all scales.

    """
    ms_seg_timestamps_list = []
    max_length = len(ms_ts_dict[base_scale]['time_stamps'])
    for data_dict in ms_ts_dict.values():
        ms_seg_timestamps = data_dict['time_stamps']
        ms_seg_timestamps = torch.tensor(ms_seg_timestamps)
        padded_ms_seg_ts = F.pad(input=ms_seg_timestamps, pad=(0, 0, 0, max_length - len(ms_seg_timestamps)), mode='constant', value=0)
        ms_seg_timestamps_list.append(padded_ms_seg_ts)
    ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
    return ms_seg_timestamps

def get_interpolate_weights(
    ms_seg_timestamps: torch.Tensor, 
    base_seq_len: int, 
    msdd_multiscale_args_dict: dict, 
    emb_scale_n: int, 
    msdd_scale_n: int, 
    is_integer_ts=False
    ):
    """
    Interpolate embeddings to a finer scale.

    Args:
        emb_fix (torch.Tensor): embeddings of the base scale
        ms_seg_timestamps (torch.Tensor): timestamps of the base scale
        base_seq_len (int): length of the base scale
    
    Returns:
        emb_fix (torch.Tensor): interpolated embeddings
    """
    deci = 100.0 if is_integer_ts else 1.0
    half_scale = msdd_multiscale_args_dict['scale_dict'][emb_scale_n-1][1]
    session_scale_dist_mat = get_scale_dist_mat_t(source_scale_idx=emb_scale_n-1, 
                                                  ms_seg_timestamps=ms_seg_timestamps[:, :base_seq_len, :], 
                                                  msdd_scale_n=msdd_scale_n, deci=deci)
    target_bool = (session_scale_dist_mat < half_scale)
    session_scale_dist_mat.flatten()[target_bool.flatten() == False] = half_scale
    dist_delta = (half_scale - session_scale_dist_mat.flatten()).reshape(base_seq_len, target_bool.shape[1])
    interpolated_weights = ((dist_delta ** 2).t() / torch.sum(dist_delta ** 2, dim=1).t()).t()  
    return interpolated_weights 

def get_batch_cosine_sim(ms_emb_seq: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity in batch mode.

    Args:
        ms_emb_seq (Tensor):
            Multi-scale embedding sequence.

    Returns:
        batch_cos_sim (Tensor):
            Cosine similarity values
    """
    batch_embs = ms_emb_seq.reshape(-1, ms_emb_seq.shape[-3], ms_emb_seq.shape[-1])
    batch_cos_sim = cos_similarity_batch(batch_embs, batch_embs).reshape(ms_emb_seq.shape[0], ms_emb_seq.shape[2], ms_emb_seq.shape[1], ms_emb_seq.shape[1])
    batch_cos_sim = torch.mean(batch_cos_sim, dim=1) # Average over scales: (batch_size, base scale timesteps, base scale timesteps)
    return batch_cos_sim 

class EncDecDiarLabelModel(ModelPT, ExportableEncDecModel):
    """
    Encoder decoder class for multiscale diarization decoder (MSDD). Model class creates training, validation methods for setting
    up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * msdd_model
        * speaker_model
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        model = PretrainedModelInfo(
            pretrained_model_name="diar_msdd_telephonic",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/diar_msdd_telephonic/versions/1.0.1/files/diar_msdd_telephonic.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:diar_msdd_telephonic",
        )
        result.append(model)
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an MSDD model and the specified speaker embedding model. In this init function, training and validation datasets are prepared.
        """
        self._trainer = trainer if trainer else None
        self.cfg_msdd_model = cfg
        self.encoder_infer_mode = False
        
        self._init_segmentation_info()
        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
            self.pairwise_infer = False
        else:
            self.world_size = 1
            self.pairwise_infer = True

        # This is a temporary patch: Make sure that the session length is a multiple of the window length
        if self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec[0] in [1.6, 3.2]:
            self.cfg_msdd_model.session_len_sec = 16
        elif self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec[0] in [1.5, 3.0]:
            self.cfg_msdd_model.session_len_sec = 15
        else:
            raise ValueError(f"Unknown window length {self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec[0]}")


        self._init_msdd_scales() 
        if self._trainer is not None and self.cfg_msdd_model.get('augmentor', None) is not None:
            self.augmentor = process_augmentations(self.cfg_msdd_model.augmentor)
        else:
            self.augmentor = None
        super().__init__(cfg=self.cfg_msdd_model, trainer=trainer)
        window_length_in_sec = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        if isinstance(window_length_in_sec, int) or len(window_length_in_sec) <= 1:
            raise ValueError("window_length_in_sec should be a list containing multiple segment (window) lengths")
        else:
            self.cfg_msdd_model.msdd_module.scale_n = self.cfg_msdd_model.scale_n

        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(self.cfg_msdd_model.preprocessor)
        self.frame_per_sec = int(1 / self.preprocessor._cfg.window_stride)
        self.feat_dim = self.preprocessor._cfg.features
        self.max_feat_frame_count = int(self.msdd_multiscale_args_dict["scale_dict"][0][0] * self.frame_per_sec) # 0-th scale, window length
        self.msdd = EncDecDiarLabelModel.from_config_dict(self.cfg_msdd_model.msdd_module)
        self.msdd_classifier = EncDecDiarLabelModel.from_config_dict(self.cfg_msdd_model.msdd_classifier)
        self.global_loss_ratio = self.cfg_msdd_model.get('global_loss_ratio', 300)
        self.original_audio_offsets = {}
        self.eps = 1e-3
        
        # MSDD v2 parameters
        self.encoder_infer_mode = False

        if trainer is not None:
            self._init_vad_model()
            self._init_speaker_model()
            self.add_speaker_model_config(cfg)
            self.add_vad_model_config(cfg)
            self.loss = instantiate(self.cfg_msdd_model.loss)
            self.global_loss = instantiate(self.cfg_msdd_model.global_loss)
            self.affinity_loss = AffinityLoss()
            self.alpha = self.cfg_msdd_model.loss.alpha
            self.mu =  self.cfg_msdd_model.diarizer.vad.parameters.mu 
            self.vad_thres =  self.cfg_msdd_model.diarizer.vad.parameters.vad_threshold
            self.power_p_aff = self.cfg_msdd_model.get('power_p_aff', 3)
            self.thres_aff = self.cfg_msdd_model.get('thres_aff', 0.25)
            self.mc_max_vad = self.cfg_msdd_model.get('mc_max_vad', True) 
            self.mc_audio_normalize = self.cfg_msdd_model.get('mc_audio_normalize', True)
            self.power_p = self.cfg_msdd_model.get('power_p', 2)
            self.mix_count = self.cfg_msdd_model.get('mix_count', 3)
            # self.multichannel_mixing = self.cfg_msdd_model.get('multichannel_mixing', False)
            self.multichannel_mixing = self.cfg_msdd_model.get('multichannel_mixing', True)
        else:
            self.msdd._speaker_model = EncDecSpeakerLabelModel.from_config_dict(cfg.speaker_model_cfg)
            self.msdd._vad_model = EncDecMultiClassificationModel.from_config_dict(cfg.vad_model_cfg)
            self.subsample_vad = int(self.cfg_msdd_model.diarizer.vad.parameters.shift_length_in_sec / 0.01) 
            self.multichannel_mixing = self.cfg_msdd_model.get('multichannel_mixing', True)
        self.affinity_weighting = self.cfg_msdd_model.get('affinity_weighting', True)
        self.msdd_overlap_add = self.cfg_msdd_model.get("msdd_overlap_add", True)
        self.use_1ch_from_ch_clus = self.cfg_msdd_model.get("use_1ch_from_ch_clus", True)
        if self.cfg_msdd_model.get("multichannel", None) is not None:
            self.power_p=self.cfg_msdd_model.multichannel.parameters.get("power_p", 4)
            self.mix_count=self.cfg_msdd_model.multichannel.parameters.get("mix_count", 2) 
        else:
            self.power_p=4
            self.mix_count=2
        
        # Call `self.save_hyperparameters` in modelPT.py again since cfg should contain speaker model's config.
        self.save_hyperparameters("cfg")

        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        
        self._accuracy_vad_test = MultiBinaryAccuracy()
        self._accuracy_vad_train = MultiBinaryAccuracy()
        self._accuracy_vad_valid = MultiBinaryAccuracy()

        self.time_flag = 0.0
        self.time_flag_end = 0.0

    def _init_msdd_scales(self,):
        window_length_in_sec = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters.window_length_in_sec
        self.msdd_multiscale_args_dict = self.multiscale_args_dict
        self.model_spk_num = self.cfg_msdd_model.max_num_of_spks
        if self.cfg_msdd_model.get('interpolated_scale', None) is not None:
            if self.cfg_msdd_model.interpolated_scale > 0.1:
                raise ValueError("Interpolated scale must be smaller than 0.1")
            # Use an interpolated scale so add another scale 
            self.cfg_msdd_model.scale_n = len(window_length_in_sec) + 1 # Adding one interpolated scale
            self.emb_scale_n = len(window_length_in_sec) # Scales that are extracted from the audio
            self.msdd_multiscale_args_dict['scale_dict'][self.emb_scale_n] = (self.cfg_msdd_model.interpolated_scale, self.cfg_msdd_model.interpolated_scale/2)
            self.msdd_multiscale_args_dict['multiscale_weights'] = [1.0] * (self.emb_scale_n+1)
            self.msdd_scale_n = int(self.emb_scale_n+1) if self.cfg_msdd_model.interpolated_scale is not None else int(self.emb_scale_n)
        else:
            # Only use the scales in window_length_in_sec
            self.cfg_msdd_model.scale_n = len(window_length_in_sec)
            self.emb_scale_n = self.cfg_msdd_model.scale_n
            self.msdd_scale_n = self.cfg_msdd_model.scale_n

    def setup_optimizer_param_groups(self):
        """
        Override function in ModelPT to allow for different parameter groups for the speaker model and the MSDD model.


        """
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        param_groups, known_groups = [], []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group_levels, group_cfg_levels in param_groups_cfg.items():
                retriever = attrgetter(group_levels)
                module = retriever(self)
                if module is None:
                    raise ValueError(f"{group_levels} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(group_levels)
                    new_group = {"params": module.parameters()}
                    for k, v in group_cfg_levels.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

            other_params = []
            for n, p in self.named_parameters():
                is_unknown = True
                for group in known_groups:
                    if group in n :
                        is_unknown = False
                if is_unknown:
                    other_params.append(p)

            if len(other_params):
                param_groups = [{"params": other_params}] + param_groups
        else:
            param_groups = [{"params": self.parameters()}]

        self._optimizer_param_groups = param_groups

    def add_speaker_model_config(self, cfg):
        """
        Add config dictionary of the speaker model to the model's config dictionary. This is required to
        save and load speaker model with MSDD model.

        Args:
            cfg (DictConfig): DictConfig type variable that conatains hyperparameters of MSDD model.
        """
        with open_dict(cfg):
            cfg_cp = copy.copy(self.msdd._speaker_model.cfg)
            cfg.speaker_model_cfg = cfg_cp
            del cfg.speaker_model_cfg.train_ds
            del cfg.speaker_model_cfg.validation_ds
    
    def add_vad_model_config(self, cfg):
        """
        Add config dictionary of the speaker model to the model's config dictionary. This is required to
        save and load speaker model with MSDD model.

        Args:
            cfg (DictConfig): DictConfig type variable that conatains hyperparameters of MSDD model.
        """
        with open_dict(cfg):
            cfg_cp = copy.copy(self.msdd._vad_model.cfg)
            cfg.vad_model_cfg = cfg_cp
            del cfg.vad_model_cfg.train_ds
            del cfg.vad_model_cfg.validation_ds
            
    def _init_segmentation_info(self):
        """Initialize segmentation settings: window, shift and multiscale weights.
        """
        self._diarizer_params = self.cfg_msdd_model.diarizer
        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def _init_vad_model(self):
        model_path = self.cfg_msdd_model.diarizer.vad.model_path
        self._diarizer_params = self.cfg_msdd_model.diarizer

        if not torch.cuda.is_available():
            rank_id = torch.device('cpu')
        elif self._trainer:
            rank_id = torch.device(self._trainer.global_rank)
        else:
            rank_id = None
        
        if model_path is not None and model_path.endswith('.nemo'):
            # self.msdd._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=rank_id)
            self.msdd._vad_model = EncDecMultiClassificationModel.restore_from(restore_path=model_path, map_location=rank_id)
            
            logging.info("Speaker Model restored locally from {}".format(model_path))
        else:
            raise ValueError(f"Only .nemo model is supported now.")
        self.subsample_vad = int(self.cfg_msdd_model.diarizer.vad.parameters.shift_length_in_sec / 0.01) 

    def _init_speaker_model(self):
        """
        Initialize speaker embedding model with model name or path passed through config. Note that speaker embedding model is loaded to
        `self.msdd` to enable multi-gpu and multi-node training. In addition, speaker embedding model is also saved with msdd model when
        `.ckpt` files are saved.
        """
        model_path = self.cfg_msdd_model.diarizer.speaker_embeddings.model_path
        self._diarizer_params = self.cfg_msdd_model.diarizer

        if not torch.cuda.is_available():
            rank_id = torch.device('cpu')
        elif self._trainer:
            rank_id = torch.device(self._trainer.global_rank)
        else:
            rank_id = None

        if model_path is not None and model_path.endswith('.nemo'):
            self.msdd._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=rank_id)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(model_path, map_location=rank_id)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "titanet_large"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                model_name=model_path, map_location=rank_id
            )
        
        if self.cfg_msdd_model.get("speaker_decoder", None) is not None:
            self.msdd._speaker_model_decoder = EncDecSpeakerLabelModel.from_config_dict(self.cfg_msdd_model.speaker_decoder)
            self.msdd._speaker_model.decoder.angular = True
            self.msdd._speaker_model.decoder.final = self.msdd._speaker_model_decoder.final
            
        if self._cfg.freeze_speaker_model:
            self.msdd._speaker_model.eval()

        self._speaker_params = self.cfg_msdd_model.diarizer.speaker_embeddings.parameters
    
    def __setup_dataloader_from_config(self, config):
        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=self.augmentor
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0

        dataset = AudioToSpeechMSDDTrainDataset(
            manifest_filepath=config.manifest_filepath,
            emb_dir=config.emb_dir,
            multiscale_args_dict=self.msdd_multiscale_args_dict,
            soft_label_thres=config.soft_label_thres,
            random_flip=config.random_flip,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            window_stride=self.cfg_msdd_model.preprocessor.window_stride,
            emb_batch_size=100,
            pairwise_infer=False,
            global_rank=global_rank,
            encoder_infer_mode=self.encoder_infer_mode,
        )

        self.data_collection = dataset.collection
        collate_ds = dataset
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=collate_ds.msdd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def __setup_dataloader_from_config_infer(
        self, config: DictConfig, emb_seq: dict, clus_label_dict: dict, pairwise_infer=False, mc_late_fusion: bool = False
    ):
        shuffle = config.get('shuffle', False)

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = AudioToSpeechMSDDInferDataset(
            manifest_filepath=config['manifest_filepath'],
            clus_label_dict=clus_label_dict,
            emb_seq=emb_seq,
            multiscale_args_dict=self.msdd_multiscale_args_dict,
            original_audio_offsets=self.original_audio_offsets,
            soft_label_thres=config.soft_label_thres,
            seq_eval_mode=config.seq_eval_mode,
            window_stride=self._cfg.preprocessor.window_stride,
            use_single_scale_clus=False,
            pairwise_infer=pairwise_infer,
            session_len_sec=config.session_len_sec,
            max_spks=config.num_spks,
            mc_late_fusion=mc_late_fusion,
        )

        self.data_collection = dataset.collection
        collate_ds = dataset
        collate_fn = collate_ds.msdd_infer_collate_fn
        batch_size = config['batch_size']
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_config,)

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config,)

    def segment_diar_window_test_dataset(self, test_data_config: Optional[Union[DictConfig, Dict]], overlap_add=False, longest_scale_idx=0):
        test_manifest_jsons = read_manifest(test_data_config['manifest_filepath'])
        self.segmented_manifest_list, self.uniq_id_segment_counts = [], {}
        self.multiscale_args_dict = self.msdd_multiscale_args_dict # Use interpolated scale from MSDD model
        ovl_len = self.msdd_multiscale_args_dict['scale_dict'][longest_scale_idx][0]
        shift = self.cfg.session_len_sec - 2 * ovl_len
        if overlap_add:
            shift = ovl_len 
        else:
            shift = self.cfg.session_len_sec - 2 * ovl_len
            
        self.diar_window = self.cfg.session_len_sec # v2 setting
        self.diar_window_shift = self.cfg.session_len_sec - ovl_len # v2 setting
        self.diar_ovl_len = ovl_len # v2 setting
        session_durations_list = [] 
        for manifest_json in test_manifest_jsons:
            if manifest_json['duration'] is None:
                manifest_json['duration'] = sox.file_info.duration(manifest_json['audio_filepath'])
            total_duration = float(manifest_json['duration'])
            total_offset = float(manifest_json['offset'])
            double_ovl = True 
            
            if double_ovl:
                subsegments = []
                total_chunks = int(np.ceil((total_duration - shift - ovl_len) / shift) + 1)
                for idx in range(total_chunks):
                    if self.msdd_overlap_add:
                        if idx == 0:
                            subsegments.append((total_offset, self.diar_window))
                            cursor = shift
                        elif idx == total_chunks - 1:
                            dur = total_duration - cursor
                            subsegments.append((total_offset + cursor, min(self.diar_window, total_duration - cursor)))
                        else:
                            subsegments.append((total_offset + cursor, self.diar_window))
                            cursor += shift
                    else:
                        if idx == 0:
                            subsegments.append((total_offset, self.diar_window))
                            cursor = self.diar_window
                        elif idx == total_chunks - 1:
                            dur = total_duration - (cursor - 2*ovl_len)
                            subsegments.append((total_offset + cursor - 2*ovl_len, min(self.diar_window, total_duration - (cursor - 2*ovl_len))))
                        else:
                            subsegments.append((total_offset + cursor - 2*ovl_len, self.diar_window))
                            cursor = cursor - 2*ovl_len + self.diar_window
              
                if abs((subsegments[-1][0] + subsegments[-1][1]) - (total_offset + total_duration)) > 1e-5:
                    raise ValueError("Last subsegment end time is not equal to total duration")
            session_durations_list.append(manifest_json['duration'])
            if isinstance(manifest_json['audio_filepath'], list):
                uniq_id = manifest_json['uniq_id'] # Multi-channel 
            else:
                uniq_id = get_uniqname_from_filepath(manifest_json['audio_filepath']) # single-channel
            self.original_audio_offsets[uniq_id] = total_offset
            self.uniq_id_segment_counts[uniq_id] = 0
            for (stt, dur) in subsegments:
                if dur < 0:
                    raise ValueError(f"Subsegment duration is negative: {dur}")
                segment_manifest_json = copy.deepcopy(manifest_json)
                segment_manifest_json['offset'], segment_manifest_json['duration'] = stt, dur
                self.segmented_manifest_list.append(segment_manifest_json)
                self.uniq_id_segment_counts[uniq_id] += 1
        max_dur = max(session_durations_list)
        scale_n = len(self.multiscale_args_dict['scale_dict'])
        max_len_ms_ts, ms_seg_counts = get_ms_seg_timestamps(uniq_id='[max_len]', 
                                                            offset=0, 
                                                            duration=max_dur, 
                                                            feat_per_sec=100, 
                                                            scale_n=scale_n,
                                                            multiscale_args_dict=self.multiscale_args_dict,
                                                            dtype=torch.float,
                                                            min_subsegment_duration=self.multiscale_args_dict['scale_dict'][scale_n-1][0],
                                                            )
        timestamps_in_scales = [] 
        for scale_idx in range(max_len_ms_ts.shape[0]):
            timestamps_in_scales.append(max_len_ms_ts[scale_idx][:ms_seg_counts[scale_idx]])
        max_len_scale_mapping_list = get_argmin_mat_large(timestamps_in_scales=timestamps_in_scales)
        max_len_scale_mapping = torch.stack(max_len_scale_mapping_list)
        max_len_ms_ts_rep = self.repeat_and_align(ms_seg_timestamps=max_len_ms_ts.unsqueeze(0), 
                                                  scale_mapping=max_len_scale_mapping.unsqueeze(0),
                                                  all_seq_len = ms_seg_counts[-1],
                                                  batch_size=1)
        if overlap_add:
            if '.msdd_segmented.json' not in test_data_config.manifest_filepath:
                msdd_segmented_manifest_path = test_data_config.manifest_filepath.replace('.json', '.msdd_segmented.json') 
            self.msdd_segmented_manifest_path = msdd_segmented_manifest_path
            test_data_config.manifest_filepath = msdd_segmented_manifest_path
            self.msdd_segmented_manifest_list = copy.deepcopy(self.segmented_manifest_list)
            if not os.path.exists(msdd_segmented_manifest_path):
                write_manifest(output_path=msdd_segmented_manifest_path, target_manifest=self.msdd_segmented_manifest_list)
        else:
            if '.segmented.json' not in test_data_config.manifest_filepath:
                segmented_manifest_path = test_data_config.manifest_filepath.replace('.json', '.segmented.json') 
            self.segmented_manifest_path = segmented_manifest_path
            test_data_config.manifest_filepath = segmented_manifest_path
            self.segmented_manifest_list = copy.deepcopy(self.segmented_manifest_list)
            if not os.path.exists(segmented_manifest_path):
                write_manifest(output_path=segmented_manifest_path, target_manifest=self.segmented_manifest_list)
            
            if '.msdd_segmented.json' not in test_data_config.manifest_filepath:
                msdd_segmented_manifest_path = test_data_config.manifest_filepath.replace('.segmented.json', '.msdd_segmented.json')
            self.msdd_segmented_manifest_path = msdd_segmented_manifest_path
            self.msdd_segmented_manifest_list = copy.deepcopy(self.segmented_manifest_list)
            if not os.path.exists(msdd_segmented_manifest_path):
                write_manifest(output_path=msdd_segmented_manifest_path, target_manifest=self.msdd_segmented_manifest_list)
        return test_data_config, max_len_ms_ts_rep, max_len_scale_mapping
    
    def setup_mc_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]], global_input_segmentation=True):
        if global_input_segmentation:
            test_data_config, self.max_len_ms_ts, self.max_len_scale_map = self.segment_diar_window_test_dataset(test_data_config, overlap_add=self.msdd_overlap_add)
        self._test_dl = self.__setup_dataloader_from_config_infer(
            config=test_data_config,
            emb_seq=self.emb_seq_test,
            clus_label_dict=self.clus_test_label_dict,
            mc_late_fusion=self.multi_ch_late_fusion_mode,
            pairwise_infer=self.pairwise_infer,
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]], global_input_segmentation=True):
        if global_input_segmentation:
            test_data_config, self.max_len_ms_ts, self.max_len_scale_map = self.segment_diar_window_test_dataset(test_data_config, overlap_add=self.msdd_overlap_add)
        self._test_dl = self.__setup_dataloader_from_config_infer(
            config=test_data_config,
            emb_seq=self.emb_seq_test,
            clus_label_dict=self.clus_test_label_dict,
            pairwise_infer=self.pairwise_infer,
        )
    
    def setup_encoder_infer_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        val_data_config, max_len_ms_ts, max_len_scale_map = self.segment_diar_window_test_dataset(val_data_config)
        self.encoder_infer_mode = True
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_config,)
        return max_len_ms_ts, max_len_scale_map 

    def setup_multiple_test_data(self, test_data_config):
        """
        MSDD does not use multiple_test_data template. This function is a placeholder for preventing error.
        """
        return None

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "audio_signal": NeuralType(('B', 'T'), audio_eltype),
            "audio_signal_length": NeuralType(('B',), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "ch_clus_mat": NeuralType(('B', 'C', 'C'), ProbsType()),
            "global_spk_labels": NeuralType(('B', 'T'), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
                "vad_probs": NeuralType(('B', 'T',), ProbsType()),
                "scale_weights": NeuralType(('B', 'T', 'C', 'D'), ProbsType()),
                "batch_affinity_mat": NeuralType(('B', 'T', 'T'), ProbsType()),
            }
        )

    # @check_multiscale_data
    def get_ms_emb_fixed(
        self, 
        embs: torch.Tensor, 
        scale_mapping: torch.Tensor, 
        ms_seg_counts: torch.Tensor, 
        ms_seg_timestamps: torch.Tensor,
    ):
        # check_ms_data(ms_seg_timestamps, ms_seg_counts, scale_mapping)
        batch_size = scale_mapping.shape[0]
        split_emb_tup = torch.split(embs, ms_seg_counts[:, :self.emb_scale_n].flatten().tolist(), dim=0)
        ms_ts = ms_seg_timestamps[:, :, :ms_seg_counts[0][self.emb_scale_n-1],:]

        base_seq_len = ms_seg_counts[0][self.msdd_scale_n-1].item()
        target_embs = torch.vstack(split_emb_tup).reshape(batch_size, -1, embs.shape[-1])
        intp_w = get_interpolate_weights(ms_seg_timestamps[0], 
                                         base_seq_len, 
                                         self.msdd_multiscale_args_dict, 
                                         self.emb_scale_n, 
                                         self.msdd_scale_n, 
                                         is_integer_ts=True)

        repeat_mats_ext = scale_mapping[0][:self.emb_scale_n].to(embs.device)
        all_seq_len = ms_seg_counts[0][-1].to(embs.device)
        
        # To make offset values such as, [10, 20, 60, x] -> [0, 10, 30, 90]
        ms_emb_seq = self.add_interpolated_embs(target_embs=target_embs, 
                                                intp_w=intp_w, 
                                                repeat_mats_ext=repeat_mats_ext, 
                                                ms_seg_counts=ms_seg_counts, 
                                                embs=embs, 
                                                batch_size=batch_size, 
                                                all_seq_len=all_seq_len)
        ms_ts_rep = self.repeat_and_align(ms_seg_timestamps=ms_seg_timestamps, 
                                          scale_mapping=scale_mapping, 
                                          all_seq_len=all_seq_len, 
                                          batch_size=batch_size)
        return ms_emb_seq, ms_ts_rep
    
    def repeat_and_align(
        self, 
        ms_seg_timestamps, 
        scale_mapping, 
        all_seq_len, 
        batch_size
        ):
        device = ms_seg_timestamps.device
        repeat_mats_all = scale_mapping[0].to(device)
        ms_ts = ms_seg_timestamps.reshape(batch_size, -1, 2)
        offsets_for_batch = (all_seq_len * torch.arange(self.msdd_scale_n).to(device)).unsqueeze(1).repeat(1, all_seq_len).to(device)
        repeat_mats_all = repeat_mats_all + offsets_for_batch
        ms_ts_rep = ms_ts[:, repeat_mats_all.flatten(), :].reshape(batch_size, self.msdd_scale_n, -1, 2)
        return ms_ts_rep

    def add_interpolated_embs(
        self, 
        target_embs, 
        intp_w, 
        repeat_mats_ext, 
        ms_seg_counts, 
        embs, 
        batch_size, 
        all_seq_len
        ):
        scale_count_offsets = torch.tensor([0] + torch.cumsum(ms_seg_counts[0][:self.emb_scale_n-1], dim=0).tolist())
        repeat_mats_ext = repeat_mats_ext + (scale_count_offsets.to(embs.device)).unsqueeze(1).repeat(1, all_seq_len).to(embs.device)
        extracted_embs = target_embs[:, repeat_mats_ext.flatten(), :].reshape(batch_size, self.emb_scale_n, -1, embs.shape[-1])
        finest_extracted_start = ms_seg_counts[0][:self.emb_scale_n-1].sum()
        interpolated_embs = torch.bmm(intp_w.repeat(batch_size, 1, 1), target_embs[:, finest_extracted_start:, :]).unsqueeze(1)
        ms_emb_seq = torch.cat((extracted_embs, interpolated_embs), dim=1).transpose(2, 1) 
        return ms_emb_seq
    
    @torch.no_grad()
    def get_cluster_avg_embs_model(
        self,  
        ms_emb_seq:torch.Tensor, 
        clus_label_index: torch.Tensor, 
        seq_lengths: torch.Tensor, 
        affinity_weighting: bool = False,
        add_sil_embs: bool = False,
        power_p_aff: float = 3,
        thres_aff: float = 0.5,
    ) -> torch.Tensor:
        """
        `max_base_seq_len` is the longest base scale sequence length can be used for batch processing.

        Args:
            ms_emb_seq (Tensor):
                Multi-scale embedding sequence that is mapped, matched and repeated. 
            clus_label_index (Tensor):
                Cluster label index for each segment in the batch.
            affinity_weighting (bool):
                If True, the cluster average embeddings are weighted by the purity of the cluster.
                affinity weighting is intended to filter out speaker-overlapping segments which show low purity.

        Returns:
            ms_avg_embs (Tensor):
                Cluster average embeddings for each segment in the batch.
        """
        batch_size = ms_emb_seq.shape[0]
        max_base_seq_len = torch.max(seq_lengths)

        # Create 0 and 1 to mask embedding vectors with speaker labels 
        spk_label_mask_binary = torch.stack([ (clus_label_index == spk).float() for spk  in range(self.model_spk_num) ]).permute(1, 2, 0)
        max_spk_counts = torch.max((clus_label_index.max(dim=1)[0]+1).long())
        if self.affinity_weighting:
            ms_emb_seq = ms_emb_seq.float()
            aff_sum = get_batch_cosine_sim(ms_emb_seq).sum(dim=1)
            aff_sum_weight = (1/aff_sum.max(dim=1)[0].unsqueeze(1).repeat(1, aff_sum.shape[1]))* aff_sum
            aff_sum_weight = aff_sum_weight ** self.power_p_aff
            aff_sum_weight[aff_sum_weight < self.thres_aff] = 0
            spk_label_mask = aff_sum_weight.unsqueeze(2).repeat(1,1,self.model_spk_num) * spk_label_mask_binary
        else:
            spk_label_mask = spk_label_mask_binary
        spk_label_mask_sum = spk_label_mask.sum(dim=1) 
        spk_label_mask_sum[(spk_label_mask_sum == 0)] = 1 # avoid divide by zero for empty clusters
        # ms_emb_seq should be matched with spk_label_mask's length so truncate it 
        ms_emb_seq_trunc = ms_emb_seq[:, :max_base_seq_len, :, :].reshape(batch_size, max_base_seq_len, -1).type(torch.float32)
        ms_weighted_sum_spk = torch.bmm(spk_label_mask.permute(0, 2, 1), ms_emb_seq_trunc)
        ms_weighted_sum_spk = ms_weighted_sum_spk.permute(0, 2, 1).reshape(batch_size, self.msdd_scale_n, -1, self.model_spk_num)
        denom_label_count = torch.tile((1/spk_label_mask_sum).unsqueeze(1).unsqueeze(1), (1, self.msdd_scale_n, ms_emb_seq.shape[3], 1))
        ms_avg_embs = ms_weighted_sum_spk * denom_label_count # (B, n_scales, D, n_spks)
        if max_spk_counts > 1: 
            ms_avg_vars = torch.var(ms_avg_embs[:,:,:,:max_spk_counts], dim=3)
            max_per_sample_scale = torch.var(ms_avg_embs[:,:,:,:max_spk_counts], dim=3).max(dim=2)[0]
        else:
            ms_avg_vars = torch.ones_like(torch.var(ms_avg_embs[:,:,:,:max_spk_counts], dim=3))
            max_per_sample_scale = torch.ones_like(torch.var(ms_avg_embs[:,:,:,:max_spk_counts], dim=3).max(dim=2)[0])
        norm_weights = (1/max_per_sample_scale).unsqueeze(-1).repeat(1,1,ms_avg_embs.shape[2])
        ms_avg_var_weights = (ms_avg_vars * norm_weights)
        return ms_avg_embs, ms_avg_var_weights

    def get_feature_index_map(
        self, 
        emb_scale_n,
        processed_signal, 
        ms_seg_timestamps, 
        ms_seg_counts,
        device: torch.device,
        ):
        batch_size = processed_signal.shape[0]
        ms_seg_counts_embs = ms_seg_counts[:, :emb_scale_n] 

        total_seg_count = torch.sum(ms_seg_counts_embs)
        ms_seg_counts_embs_flatten =  ms_seg_counts_embs.flatten()

        # The following index-tensors are needed for matrix reshaping without nested for-loops.
        batch_index_range = torch.repeat_interleave(torch.arange(batch_size).to(device), ms_seg_counts_embs.sum(dim=1), dim=0)
        scale_index_range = torch.repeat_interleave(torch.arange(emb_scale_n).repeat(batch_size).to(device) , ms_seg_counts_embs_flatten)

        # Pre-compute sequence indices for faster assigning: 
        seq_index_range = torch.arange(ms_seg_counts_embs_flatten.max())
        segment_index_range = torch.concat([seq_index_range[:seq_len] for seq_len in ms_seg_counts_embs_flatten]).to(device)
        target_timestamps = ms_seg_timestamps[batch_index_range, scale_index_range, segment_index_range, :].to(torch.int64)
        feature_count_range = target_timestamps[:, 1] - target_timestamps[:, 0]
        
        # Pre-compute feature indices for faster assigning:
        feature_frame_length_range, feature_frame_interval_range= self.get_feat_range_matirx(max_feat_len=processed_signal.shape[2], 
                                                                                             feature_count_range=feature_count_range, 
                                                                                             target_timestamps=target_timestamps, 
                                                                                             device=processed_signal.device)
        # Assign frame-by-frame indices for one-pass assignment without nested for-loops
        ms_seg_count_frame_range = torch.repeat_interleave(torch.arange(total_seg_count).to(device), feature_count_range)       
        batch_frame_range = torch.repeat_interleave(batch_index_range, feature_count_range)
        return total_seg_count, ms_seg_count_frame_range, feature_frame_length_range, batch_frame_range, feature_frame_interval_range, feature_count_range

    def forward_multi_decoder(
        self,
        processed_signal, 
        processed_signal_len, 
        total_seg_count,
        ms_seg_count_frame_range, 
        feature_frame_length_range, 
        batch_frame_range, 
        feature_frame_interval_range,
        feature_count_range,
        device,
        ):
        # Assign the acoustic feature values in processed_signal at once
        encoded, _ = self.msdd._speaker_model.encoder(audio_signal=processed_signal, length=processed_signal_len)
        encoded_segments = torch.zeros(total_seg_count, encoded.shape[1], self.max_feat_frame_count).to(torch.float32).to(device)
        encoded_segments[ms_seg_count_frame_range, :, feature_frame_length_range] = encoded[batch_frame_range, :, feature_frame_interval_range]
        pools, embs = self.msdd._speaker_model.decoder(encoder_output=encoded_segments, length=feature_count_range) 
        return embs, pools

    def forward_multiscale_vad(
        self,
        vad_probs_frame, 
        total_seg_count,
        ms_seg_count_frame_range, 
        feature_frame_length_range, 
        batch_frame_range, 
        feature_frame_interval_range,
        device,
        ):
        """
        Assign the acoustic feature values in processed_signal at once.
        
        Args:
        """
        vad_prob_segments = torch.zeros(total_seg_count, self.max_feat_frame_count).to(torch.float32).to(device)
        vad_prob_segments[ms_seg_count_frame_range, feature_frame_length_range] = vad_probs_frame[batch_frame_range, feature_frame_interval_range]
        return vad_prob_segments

    def forward_multiscale(
        self, 
        processed_signal, 
        processed_signal_len, 
        ms_seg_timestamps, 
        ms_seg_counts,
        vad_probs_frame=None,
        ):
        tsc, mscfr, fflr, bfr, ffir, fcr = self.get_feature_index_map(emb_scale_n=self.emb_scale_n,
                                                                      processed_signal=processed_signal, 
                                                                      ms_seg_timestamps=ms_seg_timestamps, 
                                                                      ms_seg_counts=ms_seg_counts, 
                                                                      device=processed_signal.device)

        embs, pools = self.forward_multi_decoder(processed_signal=processed_signal, 
                                            processed_signal_len=processed_signal_len, 
                                            total_seg_count=tsc,
                                            ms_seg_count_frame_range=mscfr, 
                                            feature_frame_length_range=fflr, 
                                            batch_frame_range=bfr, 
                                            feature_frame_interval_range=ffir,
                                            feature_count_range=fcr,
                                            device=processed_signal.device,
                                            )
        if vad_probs_frame is not None:
            vad_probs_steps = self.reshape_vad_frames(vad_probs_frame=vad_probs_frame, 
                                                      max_feat_len=processed_signal.shape[2], 
                                                      ms_seg_timestamps=ms_seg_timestamps, 
                                                      max_seq_len=ms_seg_counts.max())
            
            vad_prob_segments = self.forward_multiscale_vad(vad_probs_frame=vad_probs_frame,
                                                    total_seg_count=tsc,
                                                    ms_seg_count_frame_range=mscfr, 
                                                    feature_frame_length_range=fflr, 
                                                    batch_frame_range=bfr, 
                                                    feature_frame_interval_range=ffir,
                                                    device=processed_signal.device,
                                                    )
        else:
            vad_probs_steps = None 
        return embs, pools, vad_probs_steps, vad_prob_segments
    
    def get_feat_range_matirx(self, max_feat_len, feature_count_range, target_timestamps, device):
        """ 
        """
        feat_index_range = torch.arange(0, max_feat_len).to(device) 
        feature_frame_offsets = torch.repeat_interleave(target_timestamps[:, 0], feature_count_range)
        feature_frame_interval_range = torch.concat([feat_index_range[stt:end] for (stt, end) in target_timestamps]).to(device)
        feature_frame_length_range = feature_frame_interval_range - feature_frame_offsets
        return feature_frame_length_range, feature_frame_interval_range
    
    def reshape_vad_frames(self, vad_probs_frame, max_feat_len, ms_seg_timestamps, max_seq_len):
        max_seq_len = torch.min(torch.tensor([ms_seg_timestamps.shape[2], max_seq_len]))
        target_timestamps = ms_seg_timestamps[0, -1].to(torch.int64)
        feature_count_range = target_timestamps[:, 1] - target_timestamps[:, 0]
        _, ffir = self.get_feat_range_matirx(max_feat_len, feature_count_range, target_timestamps, device=ms_seg_timestamps.device)
        vad_probs_steps = vad_probs_frame[:, ffir].reshape(vad_probs_frame.shape[0], max_seq_len, -1).mean(dim=2)
        return vad_probs_steps

    def length_to_mask(self, context_embs):
        """
        Convert length values to encoder mask input tensor.

        Args:
            lengths (torch.Tensor): tensor containing lengths of sequences
            max_len (int): maximum sequence length

        Returns:
            mask (torch.Tensor): tensor of shape (batch_size, max_len) containing 0's
                                in the padded region and 1's elsewhere
        """
        lengths=torch.tensor([context_embs.shape[1]] * context_embs.shape[0]) 
        batch_size = context_embs.shape[0]
        max_len=context_embs.shape[1]
        # create a tensor with the shape (batch_size, 1) filled with ones
        row_vector = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        # create a tensor with the shape (batch_size, max_len) filled with lengths
        length_matrix = lengths.unsqueeze(1).expand(-1, max_len).to(lengths.device)
        # create a mask by comparing the row vector and length matrix
        mask = row_vector < length_matrix
        return mask.float().to(context_embs.device)

    def forward_infer(
        self, 
        ms_emb_seq,
        seq_lengths,
        ms_avg_embs,
        ):
        """

        Args:
            ms_emb_seq (torch.Tensor): tensor containing embeddings of multiscale embedding vectors
                Dimension: (batch_size, max_seg_count, msdd_scale_n, emb_dim)
            length (torch.Tensor): tensor containing lengths of multiscale segments
                Dimension: (batch_size, max_seg_count)
            ms_avg_embs (torch.Tensor): tensor containing average embeddings of multiscale segments
                Dimension: (batch_size, msdd_scale_n, emb_dim)

        """
        context_embs, scale_weights = self.msdd.forward_context(ms_emb_seq=ms_emb_seq, 
                                                                length=seq_lengths,
                                                                ms_avg_embs=ms_avg_embs)
        encoder_mask = self.length_to_mask(context_embs)
        classifier_logits = self.msdd_classifier(encoder_states=context_embs, 
                                                 encoder_mask=encoder_mask)
        preds, scale_weights = self.msdd.forward_speaker_logits(classifier_logits, scale_weights)
        return preds, scale_weights 
    
    def forward_vad(self, audio_signal, audio_signal_length):
        log_probs = self.msdd._vad_model(input_signal=audio_signal, input_signal_length=audio_signal_length)
        vad_probs = torch.softmax(log_probs, dim=-1)[:,:,1]
        vad_probs_frame = torch.repeat_interleave(vad_probs, self.subsample_vad, dim=1)
        return vad_probs_frame
    
    def mix_mc_vad(self, mc_log_probs):
        mc_vad_probs = torch.softmax(mc_log_probs, dim=-1)[:,:,:,1]
        mono_vad_probs = mc_vad_probs.max(dim=1)[0]
        vad_probs_frame = torch.repeat_interleave(mono_vad_probs, self.subsample_vad, dim=1)
        return vad_probs_frame
    
    def forward_encoder(
        self, 
        audio_signal, 
        audio_signal_length, 
        ms_seg_timestamps, 
        ms_seg_counts, 
        scale_mapping, 
        ch_clus_mat,
    ):
        """
        Encoder part for end-to-end diarizaiton model.

        """
        audio_signal = audio_signal.to(self.device)
        self.msdd._vad_model = self.msdd._vad_model.to(self.device)
        self.msdd._speaker_model = self.msdd._speaker_model.to(self.device)
        self.msdd_classifier = self.msdd_classifier.to(self.device)
        # if self.mc_audio_normalize:
        audio_signal = (1/(audio_signal.max()+self.eps)) * audio_signal 
            
        if self.multichannel_mixing and len(audio_signal.shape) > 2: # Check if audio_signal is multichannel
            audio_signal, mc_vad_logits = self._mix_to_mono(audio_signal_batch=audio_signal, 
                                                            audio_signal_len_batch=audio_signal_length, 
                                                            ch_clus_mat=ch_clus_mat, 
                                                            eval_mode=True)
        else:
            mc_vad_logits = None
            
        processed_signal, processed_signal_length = self.msdd._speaker_model.preprocessor(
            input_signal=audio_signal, length=audio_signal_length
        )
        
        if self.multichannel_mixing and self.mc_max_vad and mc_vad_logits is not None:
        # if self.multichannel_mixing:
            vad_probs_frame = self.mix_mc_vad(mc_log_probs=mc_vad_logits)
        else:
            vad_probs_frame = self.forward_vad(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        
        # preds = generate_vad_segment_table_per_tensor(vad_probs_frame[0], per_args=per_args)
        embs, pools, vad_probs, vad_probs_segments = self.forward_multiscale(
            processed_signal=processed_signal, 
            processed_signal_len=processed_signal_length, 
            ms_seg_timestamps=ms_seg_timestamps, 
            ms_seg_counts=ms_seg_counts,
            vad_probs_frame=vad_probs_frame,
        )
        
        if self._cfg.freeze_speaker_model:
            embs = embs.detach()
        
        # Reshape the embedding vectors into multi-scale inputs
        ms_emb_seq, ms_ts_rep = self.get_ms_emb_fixed(embs=embs,    
                                                      scale_mapping=scale_mapping, 
                                                      ms_seg_counts=ms_seg_counts, 
                                                      ms_seg_timestamps=ms_seg_timestamps)
        ms_pools_seq, _ = self.get_ms_emb_fixed(embs=pools,    
                                                scale_mapping=scale_mapping, 
                                                ms_seg_counts=ms_seg_counts, 
                                                ms_seg_timestamps=ms_seg_timestamps)
        ms_vad_probs, _ = self.get_ms_emb_fixed(embs=vad_probs_segments,    
                                                scale_mapping=scale_mapping, 
                                                ms_seg_counts=ms_seg_counts, 
                                                ms_seg_timestamps=ms_seg_timestamps)
        return ms_emb_seq, ms_pools_seq, ms_vad_probs, vad_probs, ms_ts_rep,
    
    def _get_vad_for_ch_merger_mat(self, audio_signal, audio_signal_len, ch_clus_mat, eval_mode=False, power_p = 4, mix_count=3):
        """
        This method returns the weights for each channel for the current batch

        Args:

        Returns:
            ch_merge_cal_mats (Tensor):
                A tensor of shape (batch_counts, 1, input_ch_n) containing the weights for each channel
        """
        if eval_mode:
            self.msdd._vad_model.eval()
        audio_signal_clus = torch.bmm(ch_clus_mat, audio_signal.transpose(2, 1))
        max_ch = audio_signal_clus.shape[1]
        audio_signal_len_batch = audio_signal_len.repeat(max_ch) 
        audio_signal_batch = audio_signal_clus.transpose(1, 2).reshape(-1, audio_signal_clus.shape[2]) 
        log_probs = self.msdd._vad_model(input_signal=audio_signal_batch, 
                                         input_signal_length=audio_signal_len_batch)
        probs = torch.softmax(log_probs, dim=-1)[:, :, 1].mean(dim=1).reshape(-1, max_ch)
        batch_weights = probs**power_p / (probs**power_p).sum(dim=1).unsqueeze(1)
        batch_weights, _ = get_top_k_for_each_row(batch_weights, k_count=mix_count, orig_dim=batch_weights.shape[1])
        batch_weights = batch_weights / torch.sum(batch_weights, dim=1, keepdim=True) 
        batch_weights = batch_weights.unsqueeze(1)
        mc_vad_logits = log_probs.reshape(ch_clus_mat.shape[0], ch_clus_mat.shape[1], -1, 2) # the last dim is binary vad softmax
        return batch_weights, mc_vad_logits

    def _mix_to_mono(
        self, 
        audio_signal_batch, 
        audio_signal_len_batch, 
        ch_clus_mat, 
        eval_mode=False
        ):
        audio_signal_clus = torch.bmm(ch_clus_mat, audio_signal_batch.transpose(2,1)).squeeze(1)
        ch_merger_mat, mc_vad_logits = self._get_vad_for_ch_merger_mat(
            audio_signal_batch, 
            audio_signal_len_batch, 
            ch_clus_mat, 
            eval_mode,
            power_p=self.power_p,
            mix_count=self.mix_count,
            )
        audio_signal = torch.bmm(ch_merger_mat, audio_signal_clus).squeeze(1)
        return audio_signal, mc_vad_logits
            
    def forward(
        self, 
        audio_signal, 
        audio_signal_length, 
        ms_seg_timestamps, 
        ms_seg_counts, 
        clus_label_index, 
        scale_mapping, 
        ch_clus_mat,
    ):
        """
        Forward pass for training.
        
        """        
        ms_emb_seq, ms_pools_seq, ms_vad_probs, vad_probs, _ = self.forward_encoder(
            audio_signal=audio_signal, 
            audio_signal_length=audio_signal_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            scale_mapping=scale_mapping,
            ch_clus_mat=ch_clus_mat,
        )
        ms_pools_seq = ms_pools_seq.mean(dim=2)
        # Step 2: Clustering for initialization
        # Compute the cosine similarity between the input and the cluster average embeddings
        batch_cos_sim = get_batch_cosine_sim(ms_emb_seq)
        
        if vad_probs is not None: # Apply VAD to clustering results
            clus_label_index[vad_probs < self.vad_thres] == -1
            
        ms_avg_embs, ms_avg_var_weights = self.get_cluster_avg_embs_model(ms_emb_seq, 
                                                                            clus_label_index, 
                                                                            ms_seg_counts[:,-1], 
                                                                            affinity_weighting=self.affinity_weighting,
                                                                            power_p_aff= self.power_p_aff,
                                                                            thres_aff = self.thres_aff,
                                                                            add_sil_embs=True)

        # Step 3: MSDD Inference
        preds, scale_weights = self.forward_infer(ms_emb_seq=ms_emb_seq, seq_lengths=ms_seg_counts[:, -1], ms_avg_embs=ms_avg_embs)
        return preds, vad_probs, scale_weights, batch_cos_sim, ms_pools_seq

    def training_step(self, batch: list, batch_idx: int):
        audio_signal, audio_signal_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts.detach()])
        preds, vad_probs, _, batch_affinity_mat, ms_pools_seq = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
            ch_clus_mat=ch_clus_mat,
        )
        spk_loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths) 
        # vad_loss = self.loss(probs=vad_probs.unsqueeze(2), labels=targets.max(dim=2)[0].unsqueeze(2))
        vad_loss = self.loss(probs=vad_probs.unsqueeze(2), labels=targets.max(dim=2)[0].unsqueeze(2),  signal_lengths=sequence_lengths)
        loss_glb = self.global_loss(logits=ms_pools_seq.reshape(-1, ms_pools_seq.shape[-1]), labels=global_spk_labels.view(-1))
        loss_1 = (1 - self.mu) * spk_loss + self.mu * vad_loss
        loss_2 = self.affinity_loss.forward(batch_affinity_mat=batch_affinity_mat, targets=targets)
        loss = (1-self.alpha) * loss_1 + self.alpha * loss_2
        # loss = (1-self.alpha) * loss_1 + self.alpha * loss_2 + self.global_loss_ratio*loss_glb
        self._accuracy_train(preds, targets, sequence_lengths)
        self._accuracy_vad_train(vad_probs.unsqueeze(2), targets.max(dim=2)[0].unsqueeze(2), sequence_lengths)
        torch.cuda.empty_cache()
        f1_acc = self._accuracy_train.compute()
        f1_vad_acc = self._accuracy_vad_train.compute()
        self.log('loss', loss, sync_dist=True)
        self.log('loss_glb', self.global_loss_ratio * loss_glb, sync_dist=True)
        self.log('loss_bce', (1-self.alpha) * loss_1, sync_dist=True)
        self.log('loss_aff', self.alpha * loss_2, sync_dist=True)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'], sync_dist=True)
        self.log('train_f1_all_acc', f1_acc, sync_dist=True)
        self.log('train_f1_vad_acc', f1_vad_acc, sync_dist=True)
        self._accuracy_train.reset()
        return {'loss': loss}

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
    # def validation_step(self, batch: list, batch_idx: int):
        audio_signal, audio_signal_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets, global_spk_labels = batch
        sequence_lengths = torch.tensor([x[-1] for x in ms_seg_counts])
        preds, vad_probs, _, batch_affinity_mat, ms_pools_seq = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            ms_seg_timestamps=ms_seg_timestamps,
            ms_seg_counts=ms_seg_counts,
            clus_label_index=clus_label_index,
            scale_mapping=scale_mapping,
            ch_clus_mat=ch_clus_mat,
        )
        # batch_global_spk_labels = clus_label_index.unsqueeze(2).repeat(1,1,ms_emb_seq.shape[2]).reshape(clus_label_index.shape[0], -1)
        loss_glb = self.global_loss(logits=ms_pools_seq.reshape(-1, ms_pools_seq.shape[-1]), labels=global_spk_labels.view(-1))
        # spk_loss = self.loss(probs=preds, labels=targets) 
        spk_loss = self.loss(probs=preds, labels=targets, signal_lengths=sequence_lengths)
        # vad_loss = self.loss(probs=vad_probs.unsqueeze(2), labels=targets.max(dim=2)[0].unsqueeze(2))
        vad_loss = self.loss(probs=vad_probs.unsqueeze(2), labels=targets.max(dim=2)[0].unsqueeze(2), signal_lengths=sequence_lengths)
        loss_1 = (1 - self.mu) * spk_loss + self.mu * vad_loss
        loss_2 = self.affinity_loss.forward(batch_affinity_mat=batch_affinity_mat, targets=targets)
        loss = (1-self.alpha) * loss_1 + self.alpha * loss_2 + self.global_loss_ratio*loss_glb
        self._accuracy_valid(preds, targets, sequence_lengths)
        self._accuracy_vad_valid(vad_probs.unsqueeze(2), targets.max(dim=2)[0].unsqueeze(2), sequence_lengths)
        f1_acc = self._accuracy_valid.compute()
        f1_vad_acc = self._accuracy_vad_valid.compute()
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_loss_glb', self.global_loss_ratio * loss_glb, sync_dist=True)
        self.log('val_f1_acc', f1_acc, sync_dist=True)
        self.log('val_f1_vad_acc', f1_vad_acc, sync_dist=True)
        self.log('val_loss_bce', (1-self.alpha) * loss_1, sync_dist=True)
        self.log('val_loss_aff',  self.alpha * loss_2, sync_dist=True)
        return {
            'val_loss': loss,
            'val_f1_acc': f1_acc,
        }

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_valid.compute()
        self._accuracy_valid.reset()

        self.log('val_loss', val_loss_mean, sync_dist=True)
        self.log('val_f1_acc', f1_acc, sync_dist=True)
        return {
            'val_loss': val_loss_mean,
            'val_f1_acc': f1_acc,
        }

    def multi_test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        f1_acc = self._accuracy_test.compute()
        self._accuracy_test.reset()
        self.log('test_f1_acc', f1_acc, sync_dist=True)
        return {
            'test_loss': test_loss_mean,
            'test_f1_acc': f1_acc,
        }

    def compute_accuracies(self):
        """
        Calculate F1 score and accuracy of the predicted sigmoid values.

        Returns:
            f1_score (float):
                F1 score of the estimated diarized speaker label sequences.
            simple_acc (float):
                Accuracy of predicted speaker labels: (total # of correct labels)/(total # of sigmoid values)
        """
        f1_score = self._accuracy_test.compute()
        num_correct = torch.sum(self._accuracy_test.true.bool())
        total_count = torch.prod(torch.tensor(self._accuracy_test.targets.shape))
        simple_acc = num_correct / total_count
        return f1_score, simple_acc

class ClusterEmbedding(torch.nn.Module):
    """
    This class is built for calculating cluster-average embeddings, segmentation and load/save of the estimated cluster labels.
    The methods in this class is used for the inference of MSDD models.

    Args:
        cfg_diar_infer (DictConfig):
            Config dictionary from diarization inference YAML file
        cfg_msdd_model (DictConfig):
            Config dictionary from MSDD model checkpoint file

    Class Variables:
        self.cfg_diar_infer (DictConfig):
            Config dictionary from diarization inference YAML file
        cfg_msdd_model (DictConfig):
            Config dictionary from MSDD model checkpoint file
        self._speaker_model (class `EncDecSpeakerLabelModel`):
            This is a placeholder for class instance of `EncDecSpeakerLabelModel`
        self.scale_window_length_list (list):
            List containing the window lengths (i.e., scale length) of each scale.
        self.emb_scale_n (int):
            Number of scales for multi-scale clustering diarizer
        self.base_scale_index (int):
            The index of the base-scale which is the shortest scale among the given multiple scales
    """

    def __init__(
        self, cfg_diar_infer: DictConfig, cfg_msdd_model: DictConfig, speaker_model: Optional[EncDecSpeakerLabelModel]
    ):
        super().__init__()
        self.cfg_diar_infer = copy.deepcopy(cfg_diar_infer)
        self._cfg_msdd = cfg_msdd_model
        self.clus_diar_model = None
        self._speaker_model = None
        self.model_spk_num = cfg_msdd_model.max_num_of_spks
        self.scale_window_length_list = list(
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec
        )
        self.emb_scale_n = len(self.scale_window_length_list)
        self.base_scale_index = len(self.scale_window_length_list) - 1
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.msdd_model = self._init_msdd_model(cfg=cfg_diar_infer, device=cfg_diar_infer.device)
        self.transfer_diar_params_to_model_params(cfg_diar_infer)
        
        # Use multi-scale settings from MSDD model config
        ms_weights = cfg_diar_infer.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.cfg_diar_infer.diarizer.speaker_embeddings.parameters = self.msdd_model.cfg_msdd_model.diarizer.speaker_embeddings.parameters
        self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.multiscale_weights = ms_weights
        self.clus_diar_model = ClusteringDiarizer(cfg=self.cfg_diar_infer, speaker_model=self.msdd_model, is_modular=False)

    def prepare_cluster_embs_infer(self, mc_input: bool = False, use_mc_embs: bool = False):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.max_num_speakers = self.cfg_diar_infer.diarizer.clustering.parameters.max_num_speakers
        emb_seq_test, ms_time_stamps, vad_probs, clus_test_label_dict = self.run_clustering_diarizer(
            self._cfg_msdd.test_ds.manifest_filepath, self._cfg_msdd.test_ds.emb_dir, mc_input=mc_input, use_mc_embs=use_mc_embs
        )
        return emb_seq_test, ms_time_stamps, vad_probs, clus_test_label_dict

    def _init_msdd_model(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig], device: str):
        """
        Initialized MSDD model with the provided config. Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        model_path = cfg.diarizer.msdd_model.model_path
        if model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {model_path}")
            msdd_model = EncDecDiarLabelModel.restore_from(restore_path=model_path, map_location=device)
        elif model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {model_path}")
            msdd_model = EncDecDiarLabelModel.load_from_checkpoint(
                checkpoint_path=model_path, map_location=device
            )
        else:
            if model_path not in get_available_model_names(EncDecDiarLabelModel):
                logging.warning(f"requested {model_path} model name not available in pretrained models, instead")
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            msdd_model = EncDecDiarLabelModel.from_pretrained(model_name=model_path, map_location=device)
        return msdd_model

    def _init_clus_diarizer(self, manifest_filepath, emb_dir):
        self.cfg_diar_infer.diarizer.manifest_filepath = manifest_filepath
        self.cfg_diar_infer.diarizer.out_dir = emb_dir

        # Run ClusteringDiarizer which includes system VAD or oracle VAD.
        self._out_dir = self.clus_diar_model._diarizer_params.out_dir
        self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(self.out_rttm_dir, exist_ok=True)

        self.clus_diar_model._cluster_params = self.cfg_diar_infer.diarizer.clustering.parameters
        self.clus_diar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clus_diar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters
        )
        self.clus_diar_model = ClusteringDiarizer(cfg=self.cfg_diar_infer, speaker_model=self.msdd_model, is_modular=False)
        
    def run_clustering_diarizer(self, manifest_filepath: str, emb_dir: str, mc_input: bool = False, use_mc_embs: bool = False):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels. Note that speaker embedding `state_dict`
        is loaded from the `state_dict` in the provided MSDD checkpoint.

        Args:
            manifest_filepath (str):
                Input manifest file for creating audio-to-RTTM mapping.
            emb_dir (str):
                Output directory where embedding files and timestamp files are saved.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing cluster-average embeddings for each session.
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        if not use_mc_embs:
            self._init_clus_diarizer(manifest_filepath, emb_dir)
        cluster_params = self.clus_diar_model._cluster_params
        cluster_params = dict(cluster_params) if isinstance(cluster_params, DictConfig) else cluster_params.dict()
        logging.info(f"Multiscale Weights: {self.clus_diar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {json.dumps(cluster_params, indent=4)}")
        if use_mc_embs:
            self.mc_embeddings, self.mc_time_stamps, self.mc_vad_probs, self.mc_session_clus_labels = self.clus_diar_model.forward_multi_channel(batch_size=self.cfg_diar_infer.batch_size)
            return (self.mc_embeddings, self.mc_time_stamps, self.mc_vad_probs, self.mc_session_clus_labels,)
        else:
            self.embeddings, self.time_stamps, self.vad_probs, self.session_clus_labels = self.clus_diar_model.forward(batch_size=self.cfg_diar_infer.batch_size, mc_input=mc_input)
            return (self.embeddings, self.time_stamps, self.vad_probs, self.session_clus_labels,)

    def transfer_diar_params_to_model_params(self, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarization inference config files
        to MSDD model config `msdd_model.cfg`.
        """
        self.msdd_model.cfg_msdd_model.diarizer.out_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.diarizer.vad = cfg.diarizer.vad

        self.msdd_model.cfg_msdd_model.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.test_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg_msdd_model.validation_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.validation_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.validation_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size

        self.msdd_model.cfg_msdd_model.max_num_of_spks = cfg.diarizer.clustering.parameters.max_num_speakers
        
        self.msdd_model.affinity_weighting = cfg.diarizer.msdd_model.parameters.affinity_weighting
        self.msdd_model.power_p_aff = cfg.diarizer.msdd_model.parameters.power_p_aff
        self.msdd_model.thres_aff = cfg.diarizer.msdd_model.parameters.thres_aff
        self.msdd_model.mc_max_vad = cfg.diarizer.multichannel.parameters.mc_max_vad
        # self.msdd_model.mc_audio_normalize = cfg.diarizer.multichannel.parameters.mc_audio_normalize
       
        self.msdd_model.power_p = cfg.diarizer.multichannel.parameters.power_p
        self.msdd_model.mix_count = cfg.diarizer.multichannel.parameters.mix_count
        
 
class NeuralDiarizer(LightningModule):
    """
    Class for inference based on multiscale diarization decoder (MSDD). MSDD requires initializing clustering results from
    clustering diarizer. Overlap-aware diarizer requires separate RTTM generation and evaluation modules to check the effect of
    overlap detection in speaker diarization.
    """

    def __init__(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig], msdd_model=None):
        super().__init__()
        self._cfg = cfg
        # Parameter settings for MSDD model
        self.use_speaker_model_from_ckpt = cfg.diarizer.msdd_model.parameters.get('use_speaker_model_from_ckpt', True)
        self.use_clus_as_main = cfg.diarizer.msdd_model.parameters.get('use_clus_as_main', False)
        self.max_overlap_spks = cfg.diarizer.msdd_model.parameters.get('max_overlap_spks', 2)
        self.num_spks_per_model = cfg.diarizer.msdd_model.parameters.get('num_spks_per_model', 2)
        self.use_adaptive_thres = cfg.diarizer.msdd_model.parameters.get('use_adaptive_thres', True)
        self.max_pred_length = cfg.diarizer.msdd_model.parameters.get('max_pred_length', 0)
        self.diar_eval_settings = cfg.diarizer.msdd_model.parameters.get(
            'diar_eval_settings', [(0.25, False), (0.25, True)]
        )
            # 'diar_eval_settings', [(0.25, True), (0.25, False), (0.0, False)]
        if msdd_model is not None:
            self.msdd_model = msdd_model
            self._speaker_model = None
        else:
            self.msdd_model = self._init_msdd_model(cfg)
        # self.clus_diar_model = self._init_clus_diarizer(cfg)
        self.diar_window_length = cfg.diarizer.msdd_model.parameters.diar_window_length
        self.feat_per_sec = 100
        
        self.transfer_diar_params_to_neural_diar_model_params(cfg)
        self.msdd_model._init_segmentation_info()

        # Initialize clustering and embedding preparation instance (as a diarization encoder).
        self.clustering_embedding = ClusterEmbedding(
            cfg_diar_infer=cfg, cfg_msdd_model=self.msdd_model.cfg, speaker_model=self._speaker_model
        )

        # Parameters for creating diarization results from MSDD outputs.
        self.clustering_max_spks = self.msdd_model._cfg.max_num_of_spks
        self.overlap_infer_spk_limit = cfg.diarizer.msdd_model.parameters.get(
            'overlap_infer_spk_limit', self.clustering_max_spks
        )
        self._mc_input = False


    def transfer_diar_params_to_neural_diar_model_params(self, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarization inference config files
        to MSDD model config `msdd_model.cfg`.
        """
        self.gamr = self._cfg.diarizer.msdd_model.parameters.global_average_mix_ratio
        self.ga_win_count = self._cfg.diarizer.msdd_model.parameters.global_average_window_count
        self.max_mc_ch_num = self._cfg.diarizer.clustering.parameters.max_mc_ch_num 
        self.multi_ch_late_fusion_mode = cfg.diarizer.msdd_model.parameters.get('multi_ch_late_fusion_mode', False)
        self.msdd_model.multi_ch_late_fusion_mode = cfg.diarizer.msdd_model.parameters.get('multi_ch_late_fusion_mode', False)
        self.use_var_weights = self._cfg.diarizer.msdd_model.parameters.use_var_weights
        # self.msdd_model.cfg_msdd_model.msdd_diar_win_shift_in_sec = self._cfg.diarizer.msdd_model.parameters.msdd_diar_win_shift_in_sec
        self.msdd_model.msdd_overlap_add = self._cfg.diarizer.msdd_model.parameters.msdd_overlap_add
        self.msdd_model.use_1ch_from_ch_clus = self._cfg.diarizer.msdd_model.parameters.use_1ch_from_ch_clus
        
        self.msdd_model.cfg_msdd_model.diarizer.out_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.test_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        
        self.msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        
        self.msdd_model.cfg_msdd_model.validation_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        self.msdd_model.cfg_msdd_model.validation_ds.emb_dir = cfg.diarizer.out_dir
        self.msdd_model.cfg_msdd_model.validation_ds.batch_size = cfg.batch_size

        self.msdd_model.cfg_msdd_model.max_num_of_spks = cfg.diarizer.clustering.parameters.max_num_speakers
       
        self.msdd_model.affinity_weighting = cfg.diarizer.msdd_model.parameters.affinity_weighting
        self.msdd_model.power_p_aff = cfg.diarizer.msdd_model.parameters.power_p_aff
        self.msdd_model.thres_aff = cfg.diarizer.msdd_model.parameters.thres_aff
        self.msdd_model.mc_max_vad = cfg.diarizer.multichannel.parameters.mc_max_vad

        self.msdd_model.power_p = cfg.diarizer.multichannel.parameters.power_p
        self.msdd_model.mix_count = cfg.diarizer.multichannel.parameters.mix_count
        

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instances (weights and configuration) into EFF archive.
        You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """
        self.clus_diar = self.clustering_embedding.clus_diar_model
        _NEURAL_DIAR_MODEL = "msdd_model.nemo"

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)
            neural_diar_model = os.path.join(tmpdir, _NEURAL_DIAR_MODEL)

            self.clus_diar.to_config_file(path2yaml_file=config_yaml)
            if self.clus_diar.has_vad_model:
                vad_model = os.path.join(tmpdir, _VAD_MODEL)
                self.clus_diar._vad_model.save_to(vad_model)
            self.clus_diar._speaker_model.save_to(spkr_model)
            self.msdd_model.save_to(neural_diar_model)
            self.clus_diar.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    def _init_msdd_model(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):
        """
        Initialized MSDD model with the provided config. 
        Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        model_path = cfg.diarizer.msdd_model.model_path
        if model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.restore_from(restore_path=model_path, map_location=cfg.device)
        elif model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.load_from_checkpoint(
                checkpoint_path=model_path, map_location=cfg.device
            )
        else:
            if model_path not in get_available_model_names(EncDecDiarLabelModel):
                logging.warning(f"requested {model_path} model name not available in pretrained models, instead")
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd_model = EncDecDiarLabelModel.from_pretrained(model_name=model_path, map_location=cfg.device)
        # Load speaker embedding model state_dict which is loaded from the MSDD checkpoint.
        self._speaker_model = None
        return self.msdd_model
    
    def check_channel_info(self):
        is_multichannel = []
        for uniq_id, manifest_dict in self.clustering_embedding.clus_diar_model.AUDIO_RTTM_MAP.items():
            if isinstance(manifest_dict['audio_filepath'], list):
                is_multichannel.append(True if len(manifest_dict['audio_filepath']) > 1 else False)
            else:
                is_multichannel.append(False)
        if any(is_multichannel):
            mc_flag = True
        else:
            mc_flag = False
        return mc_flag
    
    @torch.no_grad()
    def diarize(self, verbose: bool=True) -> Optional[List[Optional[List[Tuple[DiarizationErrorRate, Dict]]]]]:
        """
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), 
        initialization clustering and multiscale diarization decoder (MSDD).
        Note that the result of MSDD can include multiple speakers at the same time. 
        Therefore, RTTM output of MSDD needs to be based on `make_rttm_with_overlap()`
        function that can generate overlapping timestamps. 
        `self.run_overlap_aware_eval()` function performs DER evaluation.
        """
        self.msdd_model.pairwise_infer = True
        self.transfer_diar_params_to_neural_diar_model_params(self._cfg)
        # Use MSDD Multi-scale params for clustering diarizer
        self._cfg.diarizer.speaker_embeddings.parameters = self.msdd_model.cfg_msdd_model.diarizer.speaker_embeddings.parameters
        self._mc_input = self.check_channel_info()
        self.clustering_embedding._mc_input = self._mc_input
        self.msdd_model.emb_seq_test, self.msdd_model.ms_time_stamps, self.vad_probs_dict, self.msdd_model.clus_test_label_dict = self.clustering_embedding.prepare_cluster_embs_infer(mc_input=self._mc_input, use_mc_embs=False)
        if self._mc_input:
            self.msdd_model.emb_seq_test, self.msdd_model.ms_time_stamps, self.vad_probs_dict, self.msdd_model.clus_test_label_dict = self.clustering_embedding.prepare_cluster_embs_infer(mc_input=True, use_mc_embs=True)
            preds, targets, ms_ts = self.run_mc_multiscale_decoder()
        else:
            preds, targets, ms_ts = self.run_sc_multiscale_decoder()
        
        thresholds = list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold)
        for threshold in thresholds:
            outputs = self.run_overlap_aware_eval(preds, ms_ts, threshold, verbose=verbose)
        if verbose:
            self.print_configs()
        return outputs
    
    def print_configs(self):
        print(f"MSDD v2 model: {self._cfg.diarizer.msdd_model.model_path}")
        print(f"global_average_mix_ratio: {self._cfg.diarizer.msdd_model.parameters.global_average_mix_ratio}")
        print(f"diarizer.multichannel.parameters: {self._cfg.diarizer.multichannel.parameters}")
        print(f"diarizer.clustering.parameters {self._cfg.diarizer.clustering.parameters}")
        print(f"diarizer.vad.parameters {self._cfg.diarizer.vad.parameters}")
        print(f"diarizer.msdd_model.parameters {self._cfg.diarizer.msdd_model.parameters}")
        print(f"diarizer.manifest_filepath: {self._cfg.diarizer.manifest_filepath}")
    
    def collect_ms_avg_embs(self, ms_avg_embs_current, batch_uniq_ids):
        for sample_idx, uniq_id in enumerate(batch_uniq_ids):
            if uniq_id in self.ms_avg_embs_cache:
                # if len( ms_avg_embs_current[sample_idx].unsqueeze(0).shape ) == 5:
                #     max([self.ms_avg_embs_cache[uniq_id].shape[4], ms_avg_embs_current[sample_idx].unsqueeze(0).shape[4]])
                self.ms_avg_embs_cache[uniq_id] = torch.cat((self.ms_avg_embs_cache[uniq_id], ms_avg_embs_current[sample_idx].unsqueeze(0)), dim=0)
            else:
                self.ms_avg_embs_cache[uniq_id] = ms_avg_embs_current[sample_idx].unsqueeze(0)
        return self.ms_avg_embs_cache
    
    def get_average_embeddings(self, all_manifest_uniq_ids):
        for uniq_id in set(all_manifest_uniq_ids):
            avg_list = self.ms_avg_embs_cache[uniq_id]
            self.ms_avg_embs_cache[uniq_id] = torch.max(torch.stack(avg_list), dim=0)[0]
    
    def update_and_retrieve_avg_embs(self, ms_avg_embs_current, batch_uniq_ids):
        """
        """
        self._diar_window_counter = {uniq_id: 0 for uniq_id in self.ms_avg_embs_cache.keys() }
        ms_avg_embs = torch.zeros_like(ms_avg_embs_current) 
        for sample_idx, uniq_id in enumerate(batch_uniq_ids):
            max_win_count = self.ms_avg_embs_cache[uniq_id].shape[0]
            curr_idx = self._diar_window_counter[uniq_id]
            win_stt, win_end = max(0, (curr_idx-self.ga_win_count)), min(curr_idx+self.ga_win_count, max_win_count)
            global_average_context = self.ms_avg_embs_cache[uniq_id][win_stt:win_end].mean(dim=0)
            ms_avg_embs[sample_idx] = self.gamr * ms_avg_embs_current[sample_idx] + (1-self.gamr) * global_average_context
            self._diar_window_counter[uniq_id] += 1
        ms_avg_embs = ms_avg_embs.type(ms_avg_embs_current.dtype)
        return ms_avg_embs 
    
    def run_mc_multiscale_decoder(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Setup the parameters needed for batch inference and run batch inference. Note that each sample is pairwise speaker input.
        The pairwise inference results are reconstructed to make session-wise prediction results.

        ms_emb_seq: (torch.tensor)
            Dimension: (batch_size, max_seq_len, scale_n, emb_dim, max_ch)
        seq_lengths: (torch.tensor)
            Dimension: (batch_size,)
        ms_avg_embs: (torch.tensor)

        clus_label_index: (torch.tensor)
            Dimension: (batch_size, max_seq_len, max_ch)

        Returns:
            integrated_preds_list: (list)
                List containing the session-wise speaker predictions in torch.tensor format.
            targets_list: (list)
                List containing the ground-truth labels in matrix format filled with  0 or 1.
            signal_lengths_list: (list)
                List containing the actual length of each sequence in session.
        """
        self.out_rttm_dir = os.path.join(self.clustering_embedding.clus_diar_model._out_dir, 'pred_mc_rttms_with_overlap')
        self.out_json_dir = os.path.join(self.clustering_embedding.clus_diar_model._out_dir, 'pred_mc_jsons_with_overlap')
        self.msdd_model.setup_mc_test_data(self.msdd_model.cfg.test_ds, global_input_segmentation=True)
        self.msdd_model.eval()
        self.ms_avg_embs_cache = {}
        batch_size = self.msdd_model.cfg.test_ds.batch_size

        cumul_sample_count = [0]
        preds_list, targets_list, timestamps_list = [], [], []
        device = self.msdd_model.device
        all_manifest_uniq_ids = get_uniq_id_list_from_manifest(self.msdd_model.msdd_segmented_manifest_path)
        
        # Get the average embeddings for each session
        for test_batch_idx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader(), desc="Computing average embeddings")):
            mc_ms_emb_seq, _, mc_seq_lengths, mc_clus_label_index, targets = test_batch
            batch_uniq_ids = all_manifest_uniq_ids[test_batch_idx * batch_size: (test_batch_idx+1) * batch_size]
            for bi in tqdm(range(mc_ms_emb_seq.shape[0]), desc="Computing average embeddings per sample in batch", leave=False):
                mc_ms_emb_seq_bi = get_selected_channel_embs(mc_ms_emb_seq[bi], self.max_mc_ch_num, collapse_scale_dim=False)
                ms_emb_seq = mc_ms_emb_seq_bi.permute(3, 0, 1, 2)
                clus_label_index = mc_clus_label_index[bi].repeat(ms_emb_seq.shape[0], 1)
                seq_lengths = torch.max(mc_seq_lengths).repeat(ms_emb_seq.shape[0])
                ms_avg_embs_current, ms_avg_var_weights = self.msdd_model.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, seq_lengths, add_sil_embs=False)
                ms_avg_embs_current, ms_avg_var_weights = ms_avg_embs_current.to(self.msdd_model.device), ms_avg_var_weights.to(self.msdd_model.device)
                mc_ms_avg_embs = ms_avg_embs_current.permute(1,2,3,0).unsqueeze(0) # (1, scale_n, emb_dim, num_spks, max_ch) - first dim is batch
                single_uniq_id_list = [batch_uniq_ids[bi]] 
                self.ms_avg_embs_cache = self.collect_ms_avg_embs(mc_ms_avg_embs, single_uniq_id_list)
        # Run the multiscale decoder using the average embeddings (speaker profiles)
        if self._cfg.verbose:
            logging.info("Running MSDD classifier part for each session.")
        for test_batch_idx, _test_batch in enumerate(tqdm(self.msdd_model.test_dataloader(), desc="Running multiscale decoder")):
            test_batch = [ x.to(device) for x in _test_batch ]
            
            # ms_emb_seq, _, seq_lengths, clus_label_index, targets = test_batch
            mc_ms_emb_seq, _, mc_seq_lengths, mc_clus_label_index, targets = test_batch
            batch_uniq_ids = all_manifest_uniq_ids[test_batch_idx * batch_size: (test_batch_idx+1) * batch_size]
            
            # ms_emb_seq, seq_lengths, clus_label_index, targets = ms_emb_seq.to(device), seq_lengths.to(device), clus_label_index.to(device), targets.to(device)
            cumul_sample_count.append(cumul_sample_count[-1] + seq_lengths.shape[0])
            preds_batch_list = []
            for bi in tqdm(range(mc_ms_emb_seq.shape[0]), desc="Computing average embeddings per sample in batch", leave=False):
                mc_ms_emb_seq_bi = get_selected_channel_embs(mc_ms_emb_seq[bi], self.max_mc_ch_num, collapse_scale_dim=False)
                ms_emb_seq = mc_ms_emb_seq_bi.permute(3, 0, 1, 2)
                clus_label_index = mc_clus_label_index[bi].repeat(ms_emb_seq.shape[0], 1)
                seq_lengths = torch.max(mc_seq_lengths).repeat(ms_emb_seq.shape[0])
                ms_avg_embs_current, ms_avg_var_weights = self.msdd_model.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, seq_lengths, add_sil_embs=False)
                ms_avg_embs_current, ms_avg_var_weights = ms_avg_embs_current.to(self.msdd_model.device), ms_avg_var_weights.to(self.msdd_model.device)
                ms_emb_seq = ms_emb_seq.type(ms_avg_embs_current.dtype)
                single_uniq_id_list = [batch_uniq_ids[bi]] 
                mc_ms_avg_embs = ms_avg_embs_current.permute(1,2,3,0).unsqueeze(0) # (1, scale_n, emb_dim, num_spks, max_ch) - first dim is batch
                ms_avg_embs = self.update_and_retrieve_avg_embs(mc_ms_avg_embs, single_uniq_id_list)
                ms_avg_embs = ms_avg_embs.squeeze(0).permute(3, 0, 1, 2)
                
                # Apply the weights to the average embeddings and embedding sequences
                if self.use_var_weights:
                    # ms_avg_var_weights[ms_avg_var_weights < 0.5] = 0
                    # ms_avg_var_weights[ms_avg_var_weights >= 0.0] = 1.0
                    ms_avg_embs = ms_avg_embs * ms_avg_var_weights.unsqueeze(-1).repeat(1, 1, 1, ms_avg_embs.shape[-1])
                    ms_emb_seq = ms_emb_seq * ms_avg_var_weights.unsqueeze(1).repeat(1, ms_emb_seq.shape[1], 1, 1)
                    if torch.isnan(ms_avg_embs).any() or torch.isnan(ms_emb_seq).any():
                        print("ms_avg_embs has nan")
                preds, _ = self.msdd_model.forward_infer(ms_emb_seq=ms_emb_seq, 
                                                        seq_lengths=seq_lengths, 
                                                        ms_avg_embs=ms_avg_embs)
                preds_batch_list.append(preds.unsqueeze(0).permute(0, 2, 3, 1))

            preds_cat = torch.cat(preds_batch_list, dim=0) 
            preds_list.append(preds_cat.detach().cpu())
            targets_list.append(targets.detach().cpu())

        all_preds, all_targets = self._stitch_and_save(preds_list, targets_list)
        self.time_stamps = self.msdd_model.ms_time_stamps
        all_time_stamps = self.time_stamps
        return all_preds, all_targets, all_time_stamps

    
    def run_sc_multiscale_decoder(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Setup the parameters needed for batch inference and run batch inference. Note that each sample is pairwise speaker input.
        The pairwise inference results are reconstructed to make session-wise prediction results.

        Returns:
            integrated_preds_list: (list)
                List containing the session-wise speaker predictions in torch.tensor format.
            targets_list: (list)
                List containing the ground-truth labels in matrix format filled with  0 or 1.
            signal_lengths_list: (list)
                List containing the actual length of each sequence in session.
        """
        self.out_rttm_dir = os.path.join(self.clustering_embedding.clus_diar_model._out_dir, 'pred_rttms_with_overlap')
        self.out_json_dir = os.path.join(self.clustering_embedding.clus_diar_model._out_dir, 'pred_jsons_with_overlap')
        self.msdd_model.setup_test_data(self.msdd_model.cfg.test_ds, global_input_segmentation=True)
        self.msdd_model.eval()
        self.ms_avg_embs_cache = {}
        batch_size = self.msdd_model.cfg.test_ds.batch_size

        cumul_sample_count = [0]
        preds_list, targets_list, timestamps_list = [], [], []
        device = self.msdd_model.device
        all_manifest_uniq_ids = get_uniq_id_list_from_manifest(self.msdd_model.msdd_segmented_manifest_path)
        
        # Get the average embeddings for each session
        
        for test_batch_idx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader(), desc="Computing average embeddings", leave=False)):
            ms_emb_seq, _, seq_lengths, clus_label_index, targets = test_batch
            batch_uniq_ids = all_manifest_uniq_ids[test_batch_idx * batch_size: (test_batch_idx+1) * batch_size]
            # ms_avg_embs_current = self.msdd_model.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, seq_lengths, add_sil_embs=False).to(self.msdd_model.device)
            ms_avg_embs_current, ms_avg_var_weights = self.msdd_model.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, seq_lengths, add_sil_embs=False)
            ms_avg_embs_current, ms_avg_var_weights = ms_avg_embs_current.to(self.msdd_model.device), ms_avg_var_weights.to(self.msdd_model.device)
            self.collect_ms_avg_embs(ms_avg_embs_current, batch_uniq_ids)
            
        # Run the multiscale decoder using the average embeddings (speaker profiles)
        if self._cfg.verbose:
            logging.info("Running MSDD classifier part for each session.")
        for test_batch_idx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader(), desc="Running multiscale decoder")):
            ms_emb_seq, _, seq_lengths, clus_label_index, targets = test_batch
            batch_uniq_ids = all_manifest_uniq_ids[test_batch_idx * batch_size: (test_batch_idx+1) * batch_size]
            
            ms_emb_seq, seq_lengths, clus_label_index, targets = ms_emb_seq.to(device), seq_lengths.to(device), clus_label_index.to(device), targets.to(device)
            cumul_sample_count.append(cumul_sample_count[-1] + seq_lengths.shape[0])
            # ms_avg_embs_current = self.msdd_model.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, seq_lengths, add_sil_embs=False).to(self.msdd_model.device)
            ms_avg_embs_current, ms_avg_var_weights = self.msdd_model.get_cluster_avg_embs_model(ms_emb_seq, clus_label_index, seq_lengths, add_sil_embs=False)
            ms_avg_embs_current, ms_avg_var_weights = ms_avg_embs_current.to(self.msdd_model.device), ms_avg_var_weights.to(self.msdd_model.device)
            ms_emb_seq = ms_emb_seq.type(ms_avg_embs_current.dtype)
            
            ms_avg_embs = self.update_and_retrieve_avg_embs(ms_avg_embs_current, batch_uniq_ids)
            preds, _ = self.msdd_model.forward_infer(ms_emb_seq=ms_emb_seq, 
                                                     seq_lengths=seq_lengths, 
                                                     ms_avg_embs=ms_avg_embs)
             
            preds_list.append(preds.detach().cpu())
            targets_list.append(targets.detach().cpu())

        all_preds, all_targets = self._stitch_and_save(preds_list, targets_list)
        self.time_stamps = self.msdd_model.ms_time_stamps
        all_time_stamps = self.time_stamps
        return all_preds, all_targets, all_time_stamps

    def _stitch_and_save(
        self, 
        all_preds_list, 
        all_targets_list, 
        ):
        batch_size = self.msdd_model.cfg.test_ds.batch_size
        base_shift = self.msdd_model.cfg.interpolated_scale/2 
        base_window = self.msdd_model.cfg.interpolated_scale
        self.msdd_uniq_id_segment_counts = {}
        self.preds, self.targets, self.rep_counts = {}, {}, {}
        self.scale_mapping = {} 
        all_manifest_uniq_ids = get_uniq_id_list_from_manifest(self.msdd_model.msdd_segmented_manifest_path)
        ovl = int(self.msdd_model.diar_ovl_len / (self.msdd_model.cfg.interpolated_scale/2))-1
        if self.msdd_model.msdd_overlap_add:
            shift = self.msdd_model.diar_ovl_len
            shift_n = int(shift / (self.msdd_model.cfg.interpolated_scale/2))
            ovl_sec = self.msdd_model.diar_ovl_len
            ovl_add_len = int( (self.msdd_model.cfg_msdd_model.session_len_sec - shift - ovl_sec)  / (self.msdd_model.cfg.interpolated_scale/2))
            overlap_add = True
        else:
            shift = self.msdd_model.cfg_msdd_model.session_len_sec - 2*ovl
            ovl_add_len = None
            overlap_add = False

        for batch_idx, (preds, targets) in enumerate(zip(all_preds_list, all_targets_list)):
            batch_uniq_ids = all_manifest_uniq_ids[batch_idx * batch_size: (batch_idx+1) * batch_size ]
            batch_manifest = self.msdd_model.msdd_segmented_manifest_list[batch_idx * batch_size: (batch_idx+1) * batch_size ]
            for sample_id, uniq_id in enumerate(batch_uniq_ids):
                if uniq_id in self.preds:
                    preds_trunc = self.preds[uniq_id]
                    targets_trunc = self.targets[uniq_id]
                    count_trunc = self.rep_counts[uniq_id]
                else:
                    preds_trunc = None
                    targets_trunc = None
                  
                if batch_manifest[sample_id]['duration'] < self.msdd_model.cfg.session_len_sec:
                    last_trunc_index = int(np.ceil((batch_manifest[sample_id]['duration']-base_window)/base_shift))
                    preds_add = all_preds_list[batch_idx][sample_id][(ovl+1):last_trunc_index]
                    targets_add = all_targets_list[batch_idx][sample_id][(ovl+1):last_trunc_index]
                    count_add = torch.ones((preds_add.shape[0]))
                else:
                    if overlap_add and preds_trunc is not None:
                        # preds_add = all_preds_list[batch_idx][sample_id][ovl+1:-ovl] 
                        preds_add = all_preds_list[batch_idx][sample_id]
                        overlap_tgt_len = int(preds_add[ovl+1:-ovl].shape[0] - shift_n)
                        # print(f" ovl+1, ovl+overlap_tgt_len+1 {ovl+1, ovl+overlap_tgt_len+1}")
                        preds_add[(ovl+1):(ovl+overlap_tgt_len+1)] = preds_add[(ovl+1):(ovl+overlap_tgt_len+1)] + preds_trunc[-overlap_tgt_len:]
                        targets_add = all_targets_list[batch_idx][sample_id][ovl+1:-ovl]
                        count_add = torch.ones((preds_add.shape[0])) 
                        # print(f"count_trunc[-overlap_tgt_len:] min max : {count_trunc[-overlap_tgt_len:].min(), count_trunc[-overlap_tgt_len:].max()}") 
                        count_add[(ovl+1):(ovl+overlap_tgt_len+1)] = count_add[(ovl+1):(ovl+overlap_tgt_len+1)] + count_trunc[-overlap_tgt_len:]
                        preds_add = preds_add[(ovl+1):-ovl]
                        count_add = count_add[(ovl+1):-ovl]
                    else: 
                        preds_add = all_preds_list[batch_idx][sample_id][ovl+1:-ovl]
                        targets_add = all_targets_list[batch_idx][sample_id][ovl+1:-ovl]
                        count_add = torch.ones((preds_add.shape[0]))
                
                if uniq_id in self.preds:
                    if preds_add.shape[0] > 0: # If ts_add has valid length
                        self.msdd_uniq_id_segment_counts[uniq_id] += 1
                        if overlap_add and preds_trunc is not None:
                            ovl_add_len_cat = int(preds_add.shape[0] - shift_n)
                            if ovl_add_len_cat > 0:
                                self.preds[uniq_id] = torch.cat((preds_trunc[:-ovl_add_len_cat], preds_add), dim=0)
                                self.targets[uniq_id] = torch.cat((targets_trunc[:-ovl_add_len_cat], targets_add), dim=0)
                                self.rep_counts[uniq_id] = torch.cat((count_trunc[:-ovl_add_len_cat], count_add), dim=0)
                        else:
                            self.preds[uniq_id] = torch.cat((preds_trunc, preds_add), dim=0)
                            self.targets[uniq_id] = torch.cat((targets_trunc, targets_add), dim=0)
                            self.rep_counts[uniq_id] = torch.cat((count_trunc, count_add), dim=0)
                else:
                    self.msdd_uniq_id_segment_counts[uniq_id] = 1
                    self.preds[uniq_id] = all_preds_list[batch_idx][sample_id][ :-ovl]
                    self.targets[uniq_id] = all_targets_list[batch_idx][sample_id][ :-ovl]
                    self.rep_counts[uniq_id] = torch.ones((self.preds[uniq_id].shape[0]))
        
        if overlap_add:
            for uniq_id in self.preds:
                p_shape = self.preds[uniq_id].shape
                if len(p_shape) > 2: # Multi-channel Calse
                    sum_counts = self.rep_counts[uniq_id].unsqueeze(1).unsqueeze(2).repeat(1, p_shape[1], p_shape[2])
                else:
                    sum_counts = self.rep_counts[uniq_id].unsqueeze(1).repeat(1, p_shape[1])
                self.preds[uniq_id] = self.preds[uniq_id]/sum_counts
        return self.preds, self.targets

    def run_overlap_aware_eval(
        self, preds_dict, ms_ts, threshold: float, verbose: bool = False
    ) -> List[Optional[Tuple[DiarizationErrorRate, Dict]]]:
        """
        Based on the predicted sigmoid values, render RTTM files then evaluate the overlap-aware diarization results.

        Args:
            preds_list: (list)
                List containing predicted pairwise speaker labels.
            threshold: (float)
                A floating-point threshold value that determines overlapped speech detection.
                    - If threshold is 1.0, no overlap speech is detected and only detect major speaker.
                    - If threshold is 0.0, all speakers are considered active at any time step.
        """
        clus_test_label_dict = self.msdd_model.clus_test_label_dict
        logging.info(
            f"     [Threshold: {threshold:.4f}] "
        )
        outputs = []
        manifest_filepath = self._cfg.diarizer.manifest_filepath
        rttm_map = audio_rttm_map(manifest_filepath)
        all_reference, all_hypothesis = make_rttm_with_overlap(
            manifest_filepath,
            clus_label_dict=clus_test_label_dict,
            preds_dict=preds_dict,
            ms_ts=ms_ts,
            out_rttm_dir=self.out_rttm_dir,
            out_json_dir=self.out_json_dir,
            threshold=threshold,
            verbose=verbose,
            vad_params=self._cfg.diarizer.vad.parameters,
            infer_overlap=self._cfg.diarizer.msdd_model.parameters.infer_overlap,
            infer_mode=self._cfg.diarizer.msdd_model.parameters.infer_mode,
            use_ts_vad=self._cfg.diarizer.msdd_model.parameters.use_ts_vad,
            mask_spks_with_clus=self._cfg.diarizer.msdd_model.parameters.mask_spks_with_clus,
            overlap_infer_spk_limit=self._cfg.diarizer.msdd_model.parameters.overlap_infer_spk_limit,
            mc_late_fusion_mode=self._cfg.diarizer.msdd_model.parameters.mc_late_fusion_mode,
            system_name=self._cfg.diarizer.msdd_model.parameters.system_name,
            hop_len_in_cs=int(self.feat_per_sec * self.msdd_model.cfg.interpolated_scale/2),
            ts_vad_threshold=self._cfg.diarizer.msdd_model.parameters.ts_vad_threshold,
        )

        for k, (collar, ignore_overlap) in enumerate(self.diar_eval_settings):
            output = score_labels(
                rttm_map,
                all_reference,
                all_hypothesis,
                collar=collar,
                ignore_overlap=ignore_overlap,
                verbose=self._cfg.verbose,
            )
            outputs.append(output)
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vad_model_name: str = 'vad_multilingual_marblenet',
        map_location: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Instantiate a `NeuralDiarizer` to run Speaker Diarization.

        Args:
            model_name (str): Path/Name of the neural diarization model to load.
            vad_model_name (str): Path/Name of the voice activity detection (VAD) model to load.
            map_location (str): Optional str to map the instantiated model to a device (cpu, cuda).
                By default, (None), it will select a GPU if available, falling back to CPU otherwise.
            verbose (bool): Enable verbose logging when loading models/running diarization.
        Returns:
            `NeuralDiarizer`
        """
        logging.setLevel(logging.INFO if verbose else logging.WARNING)
        cfg = NeuralDiarizerInferenceConfig.init_config(
            diar_model_path=model_name, vad_model_path=vad_model_name, map_location=map_location, verbose=verbose,
        )
        return cls(cfg)

    def __call__(
        self,
        audio_filepath: str,
        batch_size: int = 64,
        num_workers: int = 1,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
        out_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> Union[Annotation, List[Annotation]]:
        """
        Run the `NeuralDiarizer` inference pipeline.

        Args:
            audio_filepath (str, list): Audio path to run speaker diarization on.
            max_speakers (int): If known, the max number of speakers in the file(s).
            num_speakers (int): If known, the exact number of speakers in the file(s).
            batch_size (int): Batch size when running inference.
            num_workers (int): Number of workers to use in data-loading.
            out_dir (str): Path to store intermediate files during inference (default temp directory).
        Returns:
            `pyannote.Annotation` for each audio path, containing speaker labels and segment timestamps.
        """
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            manifest_path = os.path.join(tmpdir, 'manifest.json')
            meta = [
                {
                    'audio_filepath': audio_filepath,
                    'offset': 0,
                    'duration': None,
                    'label': 'infer',
                    'text': '-',
                    'num_speakers': num_speakers,
                    'rttm_filepath': None,
                    'uem_filepath': None,
                }
            ]

            with open(manifest_path, 'w') as f:
                f.write('\n'.join(json.dumps(x) for x in meta))

            self._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=max_speakers,
                num_speakers=num_speakers,
                tmpdir=tmpdir,
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
            )

            self.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.diarize()

            pred_labels_clus = rttm_to_labels(f'{tmpdir}/pred_rttms/{Path(audio_filepath).stem}.rttm')
        return labels_to_pyannote_object(pred_labels_clus)

    def _initialize_configs(
        self,
        manifest_path: str,
        max_speakers: Optional[int],
        num_speakers: Optional[int],
        tmpdir: tempfile.TemporaryDirectory,
        batch_size: int,
        num_workers: int,
        verbose: bool,
    ) -> None:
        self._cfg.batch_size = batch_size
        self._cfg.num_workers = num_workers
        self._cfg.diarizer.manifest_filepath = manifest_path
        self._cfg.diarizer.out_dir = tmpdir
        self._cfg.verbose = verbose
        self._cfg.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None
        if max_speakers:
            self._cfg.diarizer.clustering.parameters.max_num_speakers = max_speakers
        self.transfer_diar_params_to_neural_diar_model_params(self._cfg)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return EncDecDiarLabelModel.list_available_models()