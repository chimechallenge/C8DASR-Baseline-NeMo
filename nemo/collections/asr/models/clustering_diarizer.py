# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union
import glob
from collections import Counter

import pandas as pd
import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics import detection
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from nemo.collections.asr.models.multi_classification_models import EncDecMultiClassificationModel
from nemo.collections.asr.parts.utils.vad_utils import (
    MultichannelVADProcessor,
    generate_vad_frame_pred,
    generate_vad_segment_table,
    prepare_manifest,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

######

import json
import os
import pickle as pkl
import shutil
import tarfile
import tempfile
from copy import deepcopy
from typing import Any, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
import torch.nn.functional as F

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.multi_classification_models import EncDecMultiClassificationModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.mixins.mixins import DiarizationMixin
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    parse_scale_configs,
    perform_clustering,
    perform_clustering_embs,
    segments_manifest_to_subsegments_manifest,
    validate_vad_manifest,
    write_rttm2manifest,
    get_uniq_id_list_from_manifest,
)
from nemo.collections.asr.parts.utils.offline_clustering import get_argmin_mat_large
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_timing_params,
    prepare_manifest,
    generate_vad_frame_pred,
    get_frame_data_from_logprob,
)



from nemo.core.classes import Model
from nemo.utils import logging, model_utils
from nemo.collections.asr.parts.utils.channel_clustering import channel_cluster_from_coherence

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['ClusteringDiarizer']

_MODEL_CONFIG_YAML = "model_config.yaml"
_VAD_MODEL = "vad_model.nemo"
_SPEAKER_MODEL = "speaker_model.nemo"


def get_available_model_names(class_name):
    "lists available pretrained model names from NGC"
    available_models = class_name.list_available_models()
    return list(map(lambda x: x.pretrained_model_name, available_models))



class ClusteringDiarizer(torch.nn.Module, Model, DiarizationMixin):
    """
    Inference model Class for offline speaker diarization. 
    This class handles required functionality for diarization : Speech Activity Detection, Segmentation, 
    Extract Embeddings, Clustering, Resegmentation and Scoring. 
    All the parameters are passed through config file 
    """

    def __init__(self, cfg: Union[DictConfig, Any], speaker_model=None, is_modular=True):
        super().__init__()
        if isinstance(cfg, DictConfig):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            # Convert config to support Hydra 1.0+ instantiation
            cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg

        # Diarizer set up
        self._diarizer_params = self._cfg.diarizer

        # init vad model
        self.has_vad_model = False
        if not self._diarizer_params.oracle_vad:
            if self._cfg.diarizer.vad.model_path is not None:
                self._vad_params = self._cfg.diarizer.vad.parameters
                self._init_vad_model()

        # init speaker model
        self.multiscale_embeddings_and_timestamps = {}
        if is_modular:
            self._init_speaker_model(speaker_model)
        else:
            self._init_diarizer_model(speaker_model)
        self._speaker_params = self._cfg.diarizer.speaker_embeddings.parameters

        # Clustering params
        self._cluster_params = self._diarizer_params.clustering.parameters
        self._argmax_ch_idx_dict = {}
        if self._diarizer_params.multi_channel_mode is not None:
            self._mc_vad_processor = MultichannelVADProcessor(cfg=cfg)
        else:
            self._mc_vad_processor = None
        
        if self._mc_vad_processor:
            self.max_ch = self._mc_vad_processor.max_ch
            self.max_clus = self._mc_vad_processor.max_clus
        else:
            self.max_ch = 1
            self.max_clus = 1
        
        self._ch_avg_mat = {}
        self._init_clus_diarizer()

    @classmethod
    def list_available_models(cls):
        pass

    def _init_vad_model(self):
        """
        Initialize VAD model with model name or path passed through config
        """
        model_path = self._cfg.diarizer.vad.model_path
        if model_path.endswith('.nemo'):
            if 'frame_vad' in model_path:
                self._vad_model = EncDecMultiClassificationModel.restore_from(restore_path=model_path, map_location=self._cfg.device)
            else:
                self._vad_model = EncDecClassificationModel.restore_from(model_path, map_location=self._cfg.device)
            logging.info("VAD model loaded locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecClassificationModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "vad_telephony_marblenet"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._vad_model = EncDecClassificationModel.from_pretrained(
                model_name=model_path, map_location=self._cfg.device
            )
        self._vad_window_length_in_sec = self._vad_params.window_length_in_sec
        self._vad_shift_length_in_sec = self._vad_params.shift_length_in_sec
        self.has_vad_model = True

    def _init_speaker_model(self, speaker_model=None):
        """
        Initialize speaker embedding model with model name or path passed through config
        """
        if speaker_model is not None:
            self._speaker_model = speaker_model
        else:
            model_path = self._cfg.diarizer.speaker_embeddings.model_path
            if model_path is not None and model_path.endswith('.nemo'):
                self._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=self._cfg.device)
                logging.info("Speaker Model restored locally from {}".format(model_path))
            elif model_path.endswith('.ckpt'):
                self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(
                    model_path, map_location=self._cfg.device
                )
                logging.info("Speaker Model restored locally from {}".format(model_path))
            else:
                if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                    logging.warning(
                        "requested {} model name not available in pretrained models, instead".format(model_path)
                    )
                    model_path = "ecapa_tdnn"
                logging.info("Loading pretrained {} model from NGC".format(model_path))
                self._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                    model_name=model_path, map_location=self._cfg.device
                )

        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def _init_diarizer_model(self, speaker_model=None):
        """
        Initialize speaker embedding model with model name or path passed through config
        """
        if speaker_model is not None:
            self._diarizer_model = speaker_model
        else:
            raise ValueError(f"Speaker model must be passed in for diarizer model.")

        self._diarizer_model.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )
        self._diarizer_model._init_msdd_scales()
        
        if self._diarizer_params.vad.parameters.use_external_vad:
            self._diarizer_model.msdd._vad_model = self._vad_model
        self.multiscale_args_dict = self._diarizer_model.multiscale_args_dict

    def _setup_vad_test_data(self, manifest_vad_input, batch_size=1):
        vad_dl_config = {
            'manifest_filepath': manifest_vad_input,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': 1,
            'vad_stream': True,
            'labels': ['infer',],
            'window_length_in_sec': self._vad_window_length_in_sec,
            'shift_length_in_sec': self._vad_shift_length_in_sec,
            'trim_silence': False,
            'num_workers': self._cfg.num_workers,
            'channel_selector': self._cfg.get('channel_selector', None),
        }
        self._vad_model.setup_test_data(test_data_config=vad_dl_config)

    def _setup_spkr_test_data(self, manifest_file):
        spk_dl_config = {
            'manifest_filepath': manifest_file,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size'),
            'trim_silence': False,
            'labels': None,
            'num_workers': self._cfg.num_workers,
            'channel_selector': self._cfg.get('channel_selector', None),
            'max_ch': self.max_ch,
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _setup_diarizer_validation_data(self, manifest_file):
        # diar_dl_config = {
        #     'manifest_filepath': manifest_file,
        #     'sample_rate': self._cfg.sample_rate,
        #     'batch_size': self._cfg.get('batch_size'),
        #     'trim_silence': False,
        #     'labels': None,
        #     'num_workers': self._cfg.num_workers,
        #     'channel_selector': self._cfg.get('channel_selector', None),
        #     'max_ch': self.max_ch,
        # }
        self._diarizer_model.cfg.validation_ds.manifest_filepath = manifest_file  
        self._diarizer_model.cfg.validation_ds.batch_size = self._cfg.batch_size
        self._diarizer_model.setup_encoder_infer_data(self._diarizer_model.cfg.validation_ds)

    def _run_vad(self, manifest_file, multi_channel=False, frame_vad=False):
        """
        Run voice activity detection. 
        Get log probability of voice activity detection and smoothes using the post processing parameters. 
        Using generated frame level predictions generated manifest file for later speaker embedding extraction.
        input:
        manifest_file (str) : Manifest file containing path to audio file and label as infer

        """
        shutil.rmtree(self._vad_dir, ignore_errors=True)
        os.makedirs(self._vad_dir)

        self._vad_model.eval()

        self._vad_dir, _ = generate_vad_frame_pred(
            vad_model=self._vad_model,
            window_length_in_sec=self._vad_params.window_length_in_sec,
            shift_length_in_sec=self._vad_params.shift_length_in_sec,
            manifest_vad_input=manifest_file,
            out_dir=self._vad_dir,
    )
        if multi_channel:
            self._vad_dir, self._argmax_ch_idx_dict, self._ch_avg_mat = self._mc_vad_processor.merge_frame_mc_vad(frame_vad_dir=self._vad_dir, 
                                                                                             audio_rttm_map=self.AUDIO_RTTM_MAP, 
                                                                                             method='max')
            self.max_clus = self._mc_vad_processor.max_clus
            print("self._argmax_ch_idx_dict", self._argmax_ch_idx_dict)
        if not self._vad_params.smoothing:
            # Shift the window by 10ms to generate the frame and use the prediction of the window to represent the label for the frame;
            self.vad_pred_dir = self._vad_dir
            frame_length_in_sec = self._vad_shift_length_in_sec
        else:
            # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
            # smoothing_method would be either in majority vote (median) or average (mean)
            logging.info("Generating predictions with overlapping input segments")
            smoothing_pred_dir = generate_overlap_vad_seq(
                frame_pred_dir=self._vad_dir,
                smoothing_method=self._vad_params.smoothing,
                overlap=self._vad_params.overlap,
                window_length_in_sec=self._vad_window_length_in_sec,
                shift_length_in_sec=self._vad_shift_length_in_sec,
                num_workers=self._cfg.num_workers,
            )
            self.vad_pred_dir = smoothing_pred_dir
            frame_length_in_sec = 0.01

        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")

        vad_params = self._vad_params if isinstance(self._vad_params, (DictConfig, dict)) else self._vad_params.dict()
        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=self.vad_pred_dir,
            postprocessing_params=vad_params,
            frame_length_in_sec=frame_length_in_sec,
            num_workers=self._cfg.num_workers,
            out_dir=self._vad_dir,
        )

        AUDIO_VAD_RTTM_MAP = {}
        for key in self.AUDIO_RTTM_MAP:
            if os.path.exists(os.path.join(table_out_dir, key + ".txt")):
                AUDIO_VAD_RTTM_MAP[key] = deepcopy(self.AUDIO_RTTM_MAP[key])
                AUDIO_VAD_RTTM_MAP[key]['rttm_filepath'] = os.path.join(table_out_dir, key + ".txt")
            else:
                logging.warning(f"no vad file found for {key} due to zero or negative duration")

        write_rttm2manifest(AUDIO_VAD_RTTM_MAP, self._vad_out_file)
        self._speaker_manifest_path = self._vad_out_file

    def _run_segmentation(self, window: float, shift: float, scale_tag: str = ''):

        self.subsegments_manifest_path = os.path.join(self._speaker_dir, f'subsegments{scale_tag}.json')
        logging.info(
            f"Subsegmentation for embedding extraction:{scale_tag.replace('_',' ')}, {self.subsegments_manifest_path}"
        )
        self.subsegments_manifest_path = segments_manifest_to_subsegments_manifest(
            segments_manifest_file=self._speaker_manifest_path,
            subsegments_manifest_file=self.subsegments_manifest_path,
            window=window,
            shift=shift,
            include_uniq_id=True,
        )
        return None

    def _perform_speech_activity_detection(self):
        """
        Checks for type of speech activity detection from config. Choices are NeMo VAD,
        external vad manifest and oracle VAD (generates speech activity labels from provided RTTM files)
        """
        if self.has_vad_model:
            self._auto_split = True
            self._split_duration = self._vad_params.get("split_duration", 500)
            manifest_vad_input = self._diarizer_params.manifest_filepath
            manifest_vad_input = manifest_vad_input

            if self._auto_split:
                logging.info("Split long audio file to avoid CUDA memory issue")
                logging.debug("Try smaller split_duration if you still have CUDA memory issue")
                config = {
                    'input': manifest_vad_input,
                    'window_length_in_sec': self._vad_window_length_in_sec,
                    'split_duration': self._split_duration,
                    'num_workers': self._cfg.num_workers,
                    'out_dir': self._diarizer_params.out_dir,
                }
                manifest_vad_input = prepare_manifest(config)
            else:
                logging.warning(
                    "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
                )

            self._setup_vad_test_data(manifest_vad_input)
            self._run_vad(manifest_vad_input, multi_channel=False, frame_vad=False)

        elif self._diarizer_params.vad.external_vad_manifest is not None:
            self._speaker_manifest_path = self._diarizer_params.vad.external_vad_manifest
        elif self._diarizer_params.oracle_vad:
            self._speaker_manifest_path = os.path.join(self._speaker_dir, 'oracle_vad_manifest.json')
            self._speaker_manifest_path = write_rttm2manifest(self.AUDIO_RTTM_MAP, self._speaker_manifest_path)
        else:
            raise ValueError(
                "Only one of diarizer.oracle_vad, vad.model_path or vad.external_vad_manifest must be passed from config"
            )
        validate_vad_manifest(self.AUDIO_RTTM_MAP, vad_manifest=self._speaker_manifest_path)

    def _get_mc_weights_per_batch(self, manifest_file):
        """
        This method returns the weights for each channel for the current batch

        Args:

        Returns:
            ch_merge_cal_mats (Tensor):
                A tensor of shape (batch_counts, 1, input_ch_n) containing the weights for each channel
        """
        manifest_list = read_manifest(manifest_file)
        # max_ch = max([x.shape[1] for key, x in self._ch_avg_mat.items()])
        uniq_id_list = [x['uniq_id'] for x in manifest_list]
        ordered_uniq_id_set = list(dict.fromkeys(uniq_id_list))
        counted_uniq_ids = Counter(uniq_id_list)
        ch_merge_cal_mats, ch_clus_mats_list = [], []
        for uniq_id in set(ordered_uniq_id_set):
            repeated_mat = self._ch_avg_mat[uniq_id].sum(dim=0).unsqueeze(0).repeat(counted_uniq_ids[uniq_id], 1).unsqueeze(1)
            repeated_ch_clus_mat = self._ch_avg_mat[uniq_id].unsqueeze(0).repeat(counted_uniq_ids[uniq_id], 1, 1)
            if repeated_mat.shape[2] < self.max_ch: # pad if the number of channels is less than max_ch
                repeated_mat = torch.nn.functional.pad(repeated_mat, (0, self.max_ch - repeated_mat.shape[2]))
            if repeated_ch_clus_mat.shape[2] < self.max_ch or repeated_ch_clus_mat.shape[1] < self.max_clus: # pad if the number of channels is less than max_ch
                repeated_ch_clus_mat = torch.nn.functional.pad(repeated_ch_clus_mat, (0, self.max_ch - repeated_ch_clus_mat.shape[2],
                                                                                      0, self.max_clus -repeated_ch_clus_mat.shape[1],
                                                                                        ))
            ch_merge_cal_mats.append(repeated_mat)
            ch_clus_mats_list.append(repeated_ch_clus_mat)
        ch_merger_mats = torch.cat(ch_merge_cal_mats, dim=0)
        ch_clus_mats = torch.cat(ch_clus_mats_list, dim=0)
        ch_merger_mats_list = torch.split(ch_merger_mats, self._cfg.batch_size, dim=0)
        ch_clus_mats_list = torch.split(ch_clus_mats, self._cfg.batch_size, dim=0)
        return ch_merger_mats_list, ch_clus_mats_list

    def _get_ch_merger_mat(self, audio_signal, audio_signal_len, ch_clus_mat):
        """
        This method returns the weights for each channel for the current batch

        Args:

        Returns:
            ch_merge_cal_mats (Tensor):
                A tensor of shape (batch_counts, 1, input_ch_n) containing the weights for each channel
        """
        self._vad_model.eval()
        audio_signal_clus = torch.bmm(ch_clus_mat, audio_signal.transpose(2, 1))
        sig_len = audio_signal_clus.shape[2]
        max_ch = audio_signal_clus.shape[1]
        audio_signal_len_batch = audio_signal_len.repeat(max_ch)
        audio_signal_batch = audio_signal_clus.transpose(1, 2).reshape(-1, sig_len)
        log_probs = self._vad_model(input_signal=audio_signal_batch, input_signal_length=audio_signal_len_batch)
        probs = torch.softmax(log_probs, dim=-1)[:, :, 1].mean(dim=1).reshape(-1, max_ch)
        batch_weights = probs**0.5 / (probs**0.5).sum(dim=1).unsqueeze(1)
        batch_weights = batch_weights.unsqueeze(1)
        return batch_weights

    def _extract_embeddings(self, manifest_file: str, scale_idx: int, num_scales: int):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use. 
        """
        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        self.embeddings = {}
        self._speaker_model.eval()
        self.time_stamps = {}

        all_embs = torch.empty([0])

        for bi, test_batch in enumerate(tqdm(
            self._speaker_model.test_dataloader(),
            desc=f'[{scale_idx+1}/{num_scales}] extract embeddings',
            leave=True,
            disable=not self.verbose,
        )):
            test_batch = [x.to(self._speaker_model.device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            if len(audio_signal.shape) > 2: # If multi-channel
                if audio_signal.shape[2] > 1:
                    audio_signal = torch.mean(audio_signal, dim=2)
                else:
                    audio_signal = audio_signal.squeeze(2)
            with autocast():
                _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.view(-1, emb_shape)
                all_embs = torch.cat((all_embs, embs.cpu().detach()), dim=0)
            del test_batch

        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                if 'uniq_id' in dic and dic['uniq_id'] is not None:
                    uniq_name = dic['uniq_id']
                else:
                    uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
                if uniq_name in self.embeddings:
                    self.embeddings[uniq_name] = torch.cat((self.embeddings[uniq_name], all_embs[i].view(1, -1)))
                else:
                    self.embeddings[uniq_name] = all_embs[i].view(1, -1)
                if uniq_name not in self.time_stamps:
                    self.time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                self.time_stamps[uniq_name].append([start, end])

        if self._speaker_params.save_embeddings:
            embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir, exist_ok=True)

            prefix = get_uniqname_from_filepath(manifest_file)
            name = os.path.join(embedding_dir, prefix)
            self._embeddings_file = name + f'_embeddings.pkl'
            pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
            logging.info("Saved embedding files to {}".format(embedding_dir))

    def path2audio_files_to_manifest(self, paths2audio_files, manifest_filepath):
        with open(manifest_filepath, 'w', encoding='utf-8') as fp:
            for audio_file in paths2audio_files:
                audio_file = audio_file.strip()
                entry = {'audio_filepath': audio_file, 'offset': 0.0, 'duration': None, 'text': '-', 'label': 'infer'}
                fp.write(json.dumps(entry) + '\n')

    def _init_clus_diarizer(self, paths2audio_files=None, batch_size=None):
        self._out_dir = self._diarizer_params.out_dir

        self._speaker_dir = os.path.join(self._diarizer_params.out_dir, 'speaker_outputs')

        if os.path.exists(self._speaker_dir):
            logging.warning("Deleting previous clustering diarizer outputs.")
            shutil.rmtree(self._speaker_dir, ignore_errors=True)
        os.makedirs(self._speaker_dir)

        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
        self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")

        if batch_size:
            self._cfg.batch_size = batch_size
        
        if paths2audio_files:
            if type(paths2audio_files) is list:
                self._diarizer_params.manifest_filepath = os.path.join(self._out_dir, 'paths2audio_filepath.json')
                self.path2audio_files_to_manifest(paths2audio_files, self._diarizer_params.manifest_filepath)
            else:
                raise ValueError("paths2audio_files must be of type list of paths to file containing audio file")

        self.AUDIO_RTTM_MAP = audio_rttm_map(self._diarizer_params.manifest_filepath)

        out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)
        return out_rttm_dir

    def _forward_speaker_encoder(self, manifest_file, batch_size=None):
        """

        ms_emb_seq has dimension: (batch_size, time_steps, scale_n, emb_dim)

        """
        self._setup_diarizer_validation_data(manifest_file)
        self._diarizer_model.msdd.eval()
        self._diarizer_model.msdd._speaker_model.eval()
        self._diarizer_model.msdd._vad_model.eval()
        self._vad_model.eval()
        self.feat_per_sec = 100
        
        self._diarizer_model.cfg.interpolated_scale
        all_embs_list, all_ts_list, all_mapping_list, all_vad_probs_list = [], [], [], []
        
        for bi, val_batch in enumerate(tqdm(
            self._diarizer_model.val_dataloader(),
            desc=f'Encoding speaker embeddings',
            leave=True,
            disable=not self.verbose,
        )):
            val_batch = [x.to(self._diarizer_model.device) for x in val_batch]
            audio_signal, audio_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, ch_clus_mat, targets = val_batch
            with autocast():
                ms_emb_seq, vad_probs, ms_ts_rep = self._diarizer_model.forward_encoder(audio_signal=audio_signal,
                                                                                        audio_signal_length=audio_length,
                                                                                        ms_seg_timestamps=ms_seg_timestamps,
                                                                                        ms_seg_counts=ms_seg_counts,
                                                                                        scale_mapping=scale_mapping, 
                                                                                        ch_clus_mat=ch_clus_mat)
                all_embs_list.append(ms_emb_seq.detach().cpu())
                all_ts_list.append(ms_ts_rep.detach().cpu())
                all_mapping_list.append(scale_mapping.detach().cpu())
                all_vad_probs_list.append(vad_probs.detach().cpu())
            
            del val_batch
            torch.cuda.empty_cache()

        embeddings, time_stamps, vad_probs, scale_mapping = self._stitch_and_save(manifest_file, all_embs_list, all_ts_list, all_mapping_list, all_vad_probs_list) 
        return embeddings, time_stamps, vad_probs, scale_mapping

    def _stitch_and_save(
        self, 
        manifest_file, 
        all_embs_list, 
        all_ts_list, 
        all_mapping_list,
        all_vad_probs_list,
        ):
        diar_manifest = read_manifest(manifest_file)
        batch_size = self._diarizer_model.cfg.validation_ds.batch_size
        longest_scale_idx = 0
        decimals = 2
        feat_per_sec = 100
        feat_len = float(1/feat_per_sec)

        overlap_sec = self._diarizer_model.diar_window - self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec[0]
        self.embeddings, self.time_stamps, self.scale_mapping, self.vad_probs = {}, {}, {}, {}
        base_shift = self._diarizer_model.cfg.interpolated_scale/2 
        base_seg_per_longest_seg = int((self._diarizer_model.diar_window - self._diarizer_model.diar_window_shift) / (self._diarizer_model.cfg.interpolated_scale/2))-1
        all_manifest_uniq_ids = get_uniq_id_list_from_manifest(self._diarizer_model.segmented_manifest_path)

        for batch_idx, (embs, time_stamps, vad_probs) in enumerate(tqdm(zip(all_embs_list, all_ts_list, all_vad_probs_list), desc="Stitching batch outputs")):
            batch_len = embs.shape[0]
            batch_uniq_ids = all_manifest_uniq_ids[batch_idx * batch_size: (batch_idx+1) * batch_size ]
            batch_manifest = self._diarizer_model.segmented_manifest_list[batch_idx * batch_size: (batch_idx+1) * batch_size ]
            for sample_id, uniq_id in enumerate(batch_uniq_ids):
                offset_feat = int(batch_manifest[sample_id]['offset']*feat_per_sec)
                last_trunc_index = int(batch_manifest[sample_id]['duration']/base_shift) 
                
                if uniq_id in self.embeddings:
                    embs_trunc = self.embeddings[uniq_id][:-base_seg_per_longest_seg, :, :]
                    ts_trunc = self.time_stamps[uniq_id][:, :-base_seg_per_longest_seg, :] 
                    vadp_trunc = self.vad_probs[uniq_id][:-base_seg_per_longest_seg]
               
                # Check if the last segment is truncated 
                if batch_manifest[sample_id]['duration'] < self._diarizer_model.cfg.session_len_sec:
                    embs_add = embs[sample_id][:last_trunc_index]
                    ts_add = time_stamps[sample_id][:, :last_trunc_index] + offset_feat
                    vadp_add = vad_probs[sample_id][:last_trunc_index]
                else:
                    embs_add = embs[sample_id]
                    ts_add = time_stamps[sample_id]+ offset_feat
                    vadp_add = vad_probs[sample_id]
                        
                if uniq_id in self.embeddings:
                    self.embeddings[uniq_id] = torch.cat((embs_trunc, embs_add), dim=0)
                    self.time_stamps[uniq_id]= torch.cat((ts_trunc, ts_add), dim=1)
                    self.vad_probs[uniq_id]= torch.cat((vadp_trunc, vadp_add), dim=0)
                else:
                    self.embeddings[uniq_id] = embs_add
                    self.time_stamps[uniq_id] = ts_add
                    self.vad_probs[uniq_id] = vadp_add
        
        global_scale_mapping = None
        for uniq_id, ms_ts in tqdm(self.time_stamps.items(), desc="Calculating scale mapping"):
            if global_scale_mapping is not None:
                self.scale_mapping[uniq_id] = global_scale_mapping
            timestamps_in_scales = []
            logging.info(f"Calculating unique timestamps in scales for {uniq_id}")
            for scale_idx in range(ms_ts.shape[0]):
                unique_ts = torch.vstack((torch.unique(ms_ts[scale_idx][:, 0]), torch.unique(ms_ts[scale_idx][:, 1]))).t()
                timestamps_in_scales.append(unique_ts)
            logging.info(f"Launching argmin_mat_large for {uniq_id}")
            global_scale_mapping = get_argmin_mat_large(timestamps_in_scales)
            self.scale_mapping[uniq_id] = global_scale_mapping
        
        logging.info(f"End of stitch_and_save: Saving embeddings to {self._speaker_dir}")
        return self.embeddings, self.time_stamps, self.vad_probs, self.scale_mapping

    def _save_embeddings(self, manifest_file):
        embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)

        prefix = get_uniqname_from_filepath(manifest_file)
        name = os.path.join(embedding_dir, prefix)
        self._embeddings_file = name + f'_embeddings.pkl'
        pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
        logging.info("Saved embedding files to {}".format(embedding_dir))
    
    def forward(self, batch_size: int = None):
        """
        Forward pass for end-to-end (including end-to-end optimized models) diarization.
        To deal with the long session audio inputs, we split the audio into smaller chunks and process.
        Each session is saved in terms of `uniq_id` in dictionary format.

        Argsu
            batch_size (int): batch size for speaker encoder

        Returns:
            clus_label_index (dict): dictionary containing uniq_id as key and list of cluster labels as value
        """
        # if self._mc_vad_processor is not None:
        #     self._perform_speech_activity_detection()
        
        embeddings, time_stamps, vad_probs, scale_mapping = self._forward_speaker_encoder(manifest_file=self._diarizer_params.manifest_filepath, batch_size=batch_size)
        session_clus_labels = self._run_clustering(embeddings, time_stamps, vad_probs, scale_mapping)
        return embeddings, time_stamps, vad_probs, session_clus_labels
    
    def _run_clustering(self, embeddings, time_stamps, vad_probs, scale_mapping):
        """
        Run clustering algorithm on embeddings and time stamps
        """
        out_rttm_dir = self._init_clus_diarizer()
        all_ref, all_hyp, uniq_clus_embs = perform_clustering_embs(
            embeddings=embeddings,
            vad_probs=vad_probs,
            time_stamps=time_stamps,
            scale_mapping=scale_mapping,
            AUDIO_RTTM_MAP=self.AUDIO_RTTM_MAP,
            out_rttm_dir=out_rttm_dir,
            clustering_params=self._cluster_params,
            verbose=self.verbose,
            vad_threshold=self._diarizer_params.vad.parameters.frame_vad_threshold,
            device=self._diarizer_model.device,
        )
                # self._diarizer_params.ignore_overlap,
        for ignore_overlap in [False, True]:
            score_labels(
                self.AUDIO_RTTM_MAP,
                all_ref,
                all_hyp,
                collar=self._diarizer_params.collar,
                ignore_overlap=ignore_overlap,
                verbose=self.verbose,
            )      
        return uniq_clus_embs


    def diarize(self, paths2audio_files: List[str] = None, batch_size: int = 0):
        """
        Diarize files provided thorugh paths2audio_files or manifest file
        input:
        paths2audio_files (List[str]): list of paths to file containing audio file
        batch_size (int): batch_size considered for extraction of speaker embeddings and VAD computation
        """
        out_rttm_dir = self._init_clus_diarizer(paths2audio_files, batch_size)

        # Speech Activity Detection
        self._perform_speech_activity_detection()

        # Segmentation
        scales = self.multiscale_args_dict['scale_dict'].items()
        for scale_idx, (window, shift) in scales:

            # Segmentation for the current scale (scale_idx)
            self._run_segmentation(window, shift, scale_tag=f'_scale{scale_idx}')

            # Embedding Extraction for the current scale (scale_idx)
            self._extract_embeddings(self.subsegments_manifest_path, scale_idx, len(scales))

            self.multiscale_embeddings_and_timestamps[scale_idx] = [self.embeddings, self.time_stamps]

        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )

        # Clustering
        all_reference, all_hypothesis = perform_clustering(
            embs_and_timestamps=embs_and_timestamps,
            AUDIO_RTTM_MAP=self.AUDIO_RTTM_MAP,
            out_rttm_dir=out_rttm_dir,
            clustering_params=self._cluster_params,
            device=self._speaker_model.device,
            verbose=self.verbose,
        )
        if True: # Don't evaluate 
            all_reference = []

        logging.info("Outputs are saved in {} directory".format(os.path.abspath(self._diarizer_params.out_dir)))

        # Scoring
        return score_labels(
            self.AUDIO_RTTM_MAP,
            all_reference,
            all_hypothesis,
            collar=self._diarizer_params.collar,
            ignore_overlap=self._diarizer_params.ignore_overlap,
            verbose=self.verbose,
        )

    @staticmethod
    def __make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            tar.add(source_dir, arcname="./")

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into EFF archive or .
         You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """

        # TODO: Why does this override the main save_to?

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)

            self.to_config_file(path2yaml_file=config_yaml)
            if self.has_vad_model:
                vad_model = os.path.join(tmpdir, _VAD_MODEL)
                self._vad_model.save_to(vad_model)
            self._speaker_model.save_to(spkr_model)
            self.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @staticmethod
    def __unpack_nemo_file(path2file: str, out_folder: str) -> str:
        if not os.path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        tar = tarfile.open(path2file, "r:gz")
        tar.extractall(path=out_folder)
        tar.close()
        return out_folder

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
    ):
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
                else:
                    config_yaml = override_config_path
                conf = OmegaConf.load(config_yaml)
                if os.path.exists(os.path.join(tmpdir, _VAD_MODEL)):
                    conf.diarizer.vad.model_path = os.path.join(tmpdir, _VAD_MODEL)
                else:
                    logging.info(
                        f'Model {cls.__name__} does not contain a VAD model. A VAD model or manifest file with'
                        f'speech segments need for diarization with this model'
                    )

                conf.diarizer.speaker_embeddings.model_path = os.path.join(tmpdir, _SPEAKER_MODEL)
                conf.restore_map_location = map_location
                OmegaConf.set_struct(conf, True)
                instance = cls(cfg=conf)

                logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                os.chdir(cwd)

        return instance

    @property
    def verbose(self) -> bool:
        return self._cfg.verbose
