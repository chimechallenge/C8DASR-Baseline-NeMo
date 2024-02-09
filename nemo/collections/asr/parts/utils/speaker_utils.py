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

import gc
import json
import math
import os
import shutil
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import omegaconf
import soundfile as sf
import torch
from pyannote.core import Annotation, Segment
from tqdm import tqdm
import torch.nn.functional as F

from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.metrics.der import get_partial_ref_labels
from nemo.collections.asr.parts.utils.online_clustering import (
    get_minimal_indices,
    stitch_cluster_labels
)

from nemo.collections.asr.parts.utils.offline_clustering import (
ts_vad_post_processing,
SpeakerClustering, 
get_argmin_mat, 
split_input_data,
cos_similarity,
cos_similarity_batch,
)
from nemo.collections.asr.parts.utils.longform_clustering import LongFormSpeakerClustering
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering, get_argmin_mat, split_input_data
from nemo.utils import logging

"""
This file contains all the utility functions required for speaker embeddings part in diarization scripts
"""


def get_uniqname_from_filepath(filepath):
    """
    Return base name from provided filepath
    """
    if type(filepath) is str:
        uniq_id = os.path.splitext(os.path.basename(filepath))[0]
        return uniq_id
    else:
        raise TypeError("input must be filepath string")


def get_uniq_id_from_manifest_line(line: str) -> str:
    """
    Retrieve `uniq_id` from the `audio_filepath` in a manifest line.
    """
    dic = json.loads(line.strip())
    if 'uniq_id' in dic and dic['uniq_id'] is not None:
        uniq_id = dic['uniq_id']
    else:
        uniq_id = get_uniqname_from_filepath(dic['audio_filepath'])
    return uniq_id


def get_uniq_id_with_dur(meta, decimals=3):
    """
    Return basename with offset and end time labels
    """
    # bare_uniq_id = get_uniqname_from_filepath(meta['audio_filepath'])
    bare_uniq_id = get_uniqname_from_filepath(meta['rttm_filepath'])
    if meta['offset'] is None and meta['duration'] is None:
        return bare_uniq_id
    if meta['offset']:
        offset = str(int(round(meta['offset'], decimals) * pow(10, decimals)))
    else:
        offset = 0
    if meta['duration']:
        endtime = str(int(round(meta['offset'] + meta['duration'], decimals) * pow(10, decimals)))
    else:
        endtime = 'NULL'
    uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
    return uniq_id


def audio_rttm_map(manifest, attach_dur=False):
    """
    This function creates AUDIO_RTTM_MAP which is used by all diarization components to extract embeddings,
    cluster and unify time stamps
    Args: manifest file that contains keys audio_filepath, rttm_filepath if exists, text, num_speakers if known and uem_filepath if exists

    returns:
    AUDIO_RTTM_MAP (dict) : A dictionary with keys of uniq id, which is being used to map audio files and corresponding rttm files
    """

    AUDIO_RTTM_MAP = {}
    with open(manifest, 'r') as inp_file:
        lines = inp_file.readlines()
        # logging.info("Number of files to diarize: {}".format(len(lines)))
        for line in lines:
            line = line.strip()
            dic = json.loads(line)

            meta = {
                'audio_filepath': dic['audio_filepath'],
                'rttm_filepath': dic.get('rttm_filepath', None),
                'offset': dic.get('offset', None),
                'duration': dic.get('duration', None),
                'text': dic.get('text', None),
                'num_speakers': dic.get('num_speakers', None),
                'uem_filepath': dic.get('uem_filepath', None),
                'ctm_filepath': dic.get('ctm_filepath', None),
            }
            if attach_dur:
                uniqname = get_uniq_id_with_dur(meta)
            else:
                if "uniq_id" in dic.keys():
                    uniqname = dic['uniq_id']
                else:
                    uniqname = get_uniqname_from_filepath(filepath=meta['audio_filepath'])

            if uniqname not in AUDIO_RTTM_MAP:
                AUDIO_RTTM_MAP[uniqname] = meta
            else:
                raise KeyError(
                    "file {} is already part of AUDIO_RTTM_MAP, it might be duplicated, Note: file basename must be unique".format(
                        meta['audio_filepath']
                    )
                )

    return AUDIO_RTTM_MAP


def parse_scale_configs(window_lengths_in_sec, shift_lengths_in_sec, multiscale_weights):
    """
    Check whether multiscale parameters are provided correctly. window_lengths_in_sec, shift_lengfhs_in_sec and
    multiscale_weights should be all provided in omegaconf.listconfig.ListConfig type. In addition, the scales
    should be provided in descending order, from the longest scale to the base scale (the shortest).

    Example:
        Single-scale setting:
            parameters.window_length_in_sec=1.5
            parameters.shift_length_in_sec=0.75
            parameters.multiscale_weights=null

        Multiscale setting (base scale - window_length 0.5 s and shift_length 0.25):
            parameters.window_length_in_sec=[1.5,1.0,0.5]
            parameters.shift_length_in_sec=[0.75,0.5,0.25]
            parameters.multiscale_weights=[1,1,1]

    In addition, you can also specify session-by-session multiscale weight. In this case, each dictionary key
    points to different weights.
    """
    check_float_config = [isinstance(var, float) for var in (window_lengths_in_sec, shift_lengths_in_sec)]
    check_list_config = [
        isinstance(var, (omegaconf.listconfig.ListConfig, list, tuple))
        for var in (window_lengths_in_sec, shift_lengths_in_sec, multiscale_weights)
    ]
    if all(check_list_config) or all(check_float_config):

        # If bare floating numbers are provided, convert them to list format.
        if all(check_float_config):
            window_lengths, shift_lengths, multiscale_weights = (
                [window_lengths_in_sec],
                [shift_lengths_in_sec],
                [1.0],
            )
        else:
            window_lengths, shift_lengths, multiscale_weights = (
                window_lengths_in_sec,
                shift_lengths_in_sec,
                multiscale_weights,
            )

        length_check = (
            len(set([len(window_lengths), len(shift_lengths), len(multiscale_weights)])) == 1
            and len(multiscale_weights) > 0
        )
        scale_order_check = (
            list(window_lengths) == sorted(window_lengths)[::-1] and list(shift_lengths) == sorted(shift_lengths)[::-1]
        )

        # Check whether window lengths are longer than shift lengths
        if len(window_lengths) > 1:
            shift_length_check = all([w > s for w, s in zip(window_lengths, shift_lengths)])
        else:
            shift_length_check = window_lengths[0] > shift_lengths[0]

        multiscale_args_dict = {'use_single_scale_clustering': False}
        if all([length_check, scale_order_check, shift_length_check]):
            if len(window_lengths) > 1:
                multiscale_args_dict['scale_dict'] = {
                    k: (w, s) for k, (w, s) in enumerate(zip(window_lengths, shift_lengths))
                }
            else:
                multiscale_args_dict['scale_dict'] = {0: (window_lengths[0], shift_lengths[0])}
            multiscale_args_dict['multiscale_weights'] = multiscale_weights
            return multiscale_args_dict
        else:
            raise ValueError('Multiscale parameters are not properly setup.')

    elif any(check_list_config):
        raise ValueError(
            'You must provide a list config for all three parameters: window, shift and multiscale weights.'
        )
    else:
        return None


def get_embs_and_timestamps(multiscale_embeddings_and_timestamps, multiscale_args_dict):
    """
    The embeddings and timestamps in multiscale_embeddings_and_timestamps dictionary are
    indexed by scale index. This function rearranges the extracted speaker embedding and
    timestamps by unique ID to make the further processing more convenient.

    Args:
        multiscale_embeddings_and_timestamps (dict):
            Dictionary of embeddings and timestamps for each scale.
        multiscale_args_dict (dict):
            Dictionary of scale information: window, shift and multiscale weights.

    Returns:
        embs_and_timestamps (dict)
            A dictionary containing embeddings and timestamps of each scale, indexed by unique ID.
    """
    embs_and_timestamps = {uniq_id: {} for uniq_id in multiscale_embeddings_and_timestamps[0][0].keys()}
    if multiscale_args_dict['use_single_scale_clustering']:
        _multiscale_args_dict = deepcopy(multiscale_args_dict)
        _multiscale_args_dict['scale_dict'] = {0: multiscale_args_dict['scale_dict'][0]}
        _multiscale_args_dict['multiscale_weights'] = multiscale_args_dict['multiscale_weights'][:1]
    else:
        _multiscale_args_dict = multiscale_args_dict

    embeddings, timestamps = multiscale_embeddings_and_timestamps[0]
    for uniq_id in embeddings.keys():
        embeddings_list, time_stamps_list, segment_index_list = [], [], []
        for scale_idx in sorted(_multiscale_args_dict['scale_dict'].keys()):
            embeddings, timestamps = multiscale_embeddings_and_timestamps[scale_idx]
            if len(embeddings[uniq_id]) != len(timestamps[uniq_id]):
                raise ValueError("Mismatch of counts between embedding vectors and timestamps")
            time_stamps_tensor = torch.tensor(timestamps[uniq_id])
            embeddings_list.append(embeddings[uniq_id])
            segment_index_list.append(embeddings[uniq_id].shape[0])
            time_stamps_list.append(time_stamps_tensor)

        embs_and_timestamps[uniq_id]['multiscale_weights'] = (
            torch.tensor(_multiscale_args_dict['multiscale_weights']).unsqueeze(0).float()
        )
        embs_and_timestamps[uniq_id]['embeddings'] = torch.cat(embeddings_list, dim=0)
        embs_and_timestamps[uniq_id]['timestamps'] = torch.cat(time_stamps_list, dim=0)
        embs_and_timestamps[uniq_id]['multiscale_segment_counts'] = torch.tensor(segment_index_list)

    return embs_and_timestamps


def get_timestamps(multiscale_timestamps, multiscale_args_dict):
    """
    The timestamps in `multiscale_timestamps` dictionary are indexed by scale index.
    This function rearranges the extracted speaker embedding and timestamps by unique ID to make the further processing more convenient.

    Args:
        multiscale_timestamps (dict):
            Dictionary of timestamps for each scale.
        multiscale_args_dict (dict):
            Dictionary of scale information: window, shift and multiscale weights.

    Returns:
        timestamps_dict (dict)
            A dictionary containing embeddings and timestamps of each scale, indexed by unique ID.
    """
    timestamps_dict = {uniq_id: {'scale_dict': {}} for uniq_id in multiscale_timestamps[0].keys()}
    for scale_idx in sorted(multiscale_args_dict['scale_dict'].keys()):
        time_stamps = multiscale_timestamps[scale_idx]
        for uniq_id in time_stamps.keys():
            timestamps_dict[uniq_id]['scale_dict'][scale_idx] = {
                'time_stamps': time_stamps[uniq_id],
            }

    return timestamps_dict


def get_contiguous_stamps(stamps):
    """
    Return contiguous time stamps
    """
    lines = deepcopy(stamps)
    if len(lines) == 0:
        return []
    contiguous_stamps = []
    for i in range(len(lines) - 1):
        start, end, speaker = lines[i].split()
        next_start, next_end, next_speaker = lines[i + 1].split()
        if float(end) > float(next_start):
            avg = str((float(next_start) + float(end)) / 2.0)
            lines[i + 1] = ' '.join([avg, next_end, next_speaker])
            contiguous_stamps.append(start + " " + avg + " " + speaker)
        else:
            contiguous_stamps.append(start + " " + end + " " + speaker)
    start, end, speaker = lines[-1].split()
    contiguous_stamps.append(start + " " + end + " " + speaker)
    return contiguous_stamps


def merge_stamps(lines):
    """
    Merge time stamps of the same speaker.
    """
    if len(lines) == 0:
        return []
    stamps = deepcopy(lines)
    overlap_stamps = []
    for i in range(len(stamps) - 1):
        start, end, speaker = stamps[i].split()
        next_start, next_end, next_speaker = stamps[i + 1].split()
        if float(end) == float(next_start) and speaker == next_speaker:
            stamps[i + 1] = ' '.join([start, next_end, next_speaker])
        else:
            overlap_stamps.append(start + " " + end + " " + speaker)
    start, end, speaker = stamps[-1].split()
    overlap_stamps.append(start + " " + end + " " + speaker)
    return overlap_stamps


def labels_to_pyannote_object(labels, uniq_name=''):
    """
    Convert the given labels to pyannote object to calculate DER and for visualization
    """
    annotation = Annotation(uri=uniq_name)
    if len(labels) == 0:
        return annotation
        # raise ValueError(f"No labels found in labels: {labels}")
    for label in labels:
        start, end, speaker = label.strip().split()
        start, end = float(start), float(end)
        annotation[Segment(start, end)] = speaker

    return annotation


def labels_to_rttmfile(labels, uniq_id, out_rttm_dir):
    """
    Write rttm file with uniq_id name in out_rttm_dir with timestamps in labels
    """
    filename = os.path.join(out_rttm_dir, uniq_id + '.rttm')
    with open(filename, 'w') as f:
        for line in labels:
            line = line.strip()
            start, end, speaker = line.split()
            duration = float(end) - float(start)
            start = float(start)
            log = 'SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(uniq_id, start, duration, speaker)
            f.write(log)

    return filename


def string_to_float(x, round_digits):
    """
    Convert string to float then round the number.
    """
    return round(float(x), round_digits)


def convert_rttm_line(rttm_line, round_digits=3):
    """
    Convert a line in RTTM file to speaker label, start and end timestamps.

    Args:
        rttm_line (str):
            A line in RTTM formatted file containing offset and duration of each segment.
        round_digits (int):
            Number of digits to be rounded.

    Returns:
        start (float)
            Start timestamp in floating point number.
        end (float):
            End timestamp in floating point number.
        speaker (str):
            speaker string in RTTM lines.
    """
    rttm = rttm_line.strip().split()
    start = string_to_float(rttm[3], round_digits)
    end = string_to_float(rttm[4], round_digits) + string_to_float(rttm[3], round_digits)
    speaker = rttm[7]
    return start, end, speaker


def rttm_to_labels(rttm_filename):
    """
    Prepare time stamps label list from rttm file
    """
    labels = []
    with open(rttm_filename, 'r') as f:
        for line in f.readlines():
            start, end, speaker = convert_rttm_line(line, round_digits=3)
            labels.append('{} {} {}'.format(start, end, speaker))
    return labels


def write_cluster_labels(base_scale_idx, lines_cluster_labels, out_rttm_dir):
    """
    Write cluster labels that are generated from clustering into a file.
    Args:
        base_scale_idx (int): The base scale index which is the highest scale index.
        lines_cluster_labels (list): The start and end time-stamps of each segment with the predicted cluster label.
        out_rttm_dir (str): The path where output rttm files are saved.
    """
    out_label_name = os.path.join(
        out_rttm_dir, '../speaker_outputs', f'subsegments_scale{base_scale_idx}_cluster.label'
    )
    with open(out_label_name, 'w') as f:
        for clus_label_line in lines_cluster_labels:
            f.write(clus_label_line)


def generate_cluster_labels(segment_ranges: List[str], cluster_labels: List[int]):
    """
    Generate cluster (speaker labels) from the segment_range list and cluster label list.

    Args:
        segment_ranges (list):
            List containing intervals (start and end timestapms, ranges) of each segment
        cluster_labels (list):
            List containing a cluster label sequence

    Returns:
        diar_hyp (list):
            List containing merged speaker-turn-level timestamps and labels in string format
            Example:
                >>>  diar_hyp = ['0.0 4.375 speaker_1', '4.375 5.125 speaker_0', ...]

        lines (list)
            List containing raw segment-level timestamps and labels in raw digits
                >>>  diar_hyp = ['0.0 0.25 speaker_1', '0.25 0.5 speaker_1', ..., '4.125 4.375 speaker_1']
    """
    lines = []
    for idx, label in enumerate(cluster_labels):
        tag = 'speaker_' + str(label)
        stt, end = segment_ranges[idx]
        lines.append(f"{stt} {end} {tag}")
    cont_lines = get_contiguous_stamps(lines)
    diar_hyp = merge_stamps(cont_lines)
    return diar_hyp, lines
                
def divide_and_conquer_clustering(ms_silsp_embs, cluster_labels_infer, unit_clus_len, max_num_speakers, base_scale_idx, sync_score_thres=0.75):
    """
    For long form audio files, perform divide and conquer clustering to get fine-grained speaker labels.

    Args:
        ms_silsp_embs (_type_): _description_
        cluster_labels_infer (_type_): _description_
        unit_clus_len (_type_): _description_
        base_scale_idx (_type_): _description_
        sync_score_thres (float, optional): _description_. Defaults to 0.75.

    Returns:
        _type_: _description_
    """
    fine_grained_scale_idx = min(ms_silsp_embs.shape[1]-1, base_scale_idx+1)
    cluster_labels_infer  = cluster_labels_infer.cuda()
    ms_silsp_embs = ms_silsp_embs[:, :(fine_grained_scale_idx+1)].cuda()
    vad_ms_emb_seq =  ms_silsp_embs[cluster_labels_infer > -1]
    ms_emb_seq = torch.split(vad_ms_emb_seq, unit_clus_len, dim=0)
    vad_cluster_labels_infer = cluster_labels_infer[cluster_labels_infer > -1]
    clus_label_index = torch.split(vad_cluster_labels_infer, unit_clus_len, dim=0)
    batch_size = len(ms_emb_seq)
    total_fine_grained_labels = []
    for sample_id in tqdm(range(batch_size), desc='Fine-grained clustering'):
        sample_ms_emb_seq = ms_emb_seq[sample_id]
        vad_mask = clus_label_index[sample_id] > -1
        num_speakers = int(clus_label_index[sample_id].max().item() + 1)
        speaker_clustering = SpeakerClustering(cuda=True)
        _cluster_labels = speaker_clustering.forward_embs(
            embs=sample_ms_emb_seq[vad_mask].mean(dim=1),
            max_num_speakers=max_num_speakers,
            oracle_num_speakers=int(num_speakers),
            max_rp_threshold= 0.05,
            use_drop_and_recluster=False,
        )
        # Resolve permuations
        offset = clus_label_index[sample_id][vad_mask].long().min()
        clus_label_vad = get_minimal_indices(clus_label_index[sample_id][vad_mask].long())
        new_label_index = stitch_cluster_labels(Y_old=clus_label_vad, Y_new=_cluster_labels.long())
        new_label_index = new_label_index.type(clus_label_vad.dtype).to(clus_label_vad.device)
        
        # If local clustering shows too much difference from global clustering, use global clustering
        sync_score =  ((clus_label_vad == new_label_index).sum() / clus_label_vad.shape[0]).item()
        logging.info(f"[Speaker Clustering] Fine grained label sync score: [{sync_score:.4f} , offset: {offset} sync_score_thres: {sync_score_thres:.3f}]")
        if sync_score < sync_score_thres:
            new_label_index = clus_label_vad + offset
        
        total_fine_grained_labels.append(new_label_index)
    
    vad_fine_grained_labels = torch.cat(total_fine_grained_labels, dim=0).to(cluster_labels_infer.device)
    fine_grained_labels = (torch.ones_like(cluster_labels_infer) * -1).to(cluster_labels_infer.device)
    fine_grained_labels[cluster_labels_infer > -1] = vad_fine_grained_labels.type(cluster_labels_infer.dtype)
    return fine_grained_labels
               
def get_cluster_labels_infer(
    ms_silsp_embs: torch.Tensor,
    cluster_labels: torch.Tensor,
    vad_decision_scaled: torch.Tensor,
    vad_decision_base: torch.Tensor,
    scale_map: torch.Tensor,
    base_scale_idx: int,
    ):


    # Convert cluster labels to the finest scale
    clus_labels_infer_org_scale = torch.zeros(ms_silsp_embs.shape[0]+1)
    clus_labels_infer_org_scale[:vad_decision_scaled.shape[0]][vad_decision_scaled] = (cluster_labels + 1).float().to(ms_silsp_embs.device)
    clus_labels_infer_org_scale -= 1

    # Convert clustering to the finest scale (=base scale)
    cluster_labels_infer = -1 * torch.ones(scale_map[0].shape[0]) # Finest scale thus, the longest vector
    max_scm = scale_map[base_scale_idx].shape[0]
    
    cluster_labels_infer[scale_map[-1][vad_decision_base[:max_scm]]]= clus_labels_infer_org_scale[scale_map[base_scale_idx][vad_decision_base[:max_scm]]]
    return cluster_labels_infer, max_scm

def get_ms_embs_and_ts(
    base_scale_idx: int,
    embeddings: torch.Tensor,
    time_stamps: torch.Tensor,
    scale_map: torch.Tensor,
    vad_probs: torch.Tensor,
    vad_threshold: float,
    feat_per_sec: int = 100,
    ):
    """
    Get multi-scale embeddings and time-stamps and perform VAD masking.

    Args:
        base_scale_idx (int):
            The base scale index which is the highest scale index.
        embeddings (torch.Tensor):
            The embeddings of the audio file.
        time_stamps (torch.Tensor):
            The time-stamps of the audio file.
        scale_map (torch.Tensor):
            The scale mapping of the audio file.
        vad_probs (torch.Tensor):
            The VAD probabilities of the audio file.
        vad_threshold (float):
            The VAD threshold to be used for VAD masking.

    Returns:
        ms_silsp_embs (torch.Tensor):
            The multi-scale embeddings of the audio file.
        ms_embs_scaled_vadmasked (torch.Tensor):
            The multi-scale embeddings of the audio file after VAD masking.
        ms_ts_scaled (torch.Tensor):
            The multi-scale time-stamps of the audio file.
        vad_decision_scaled (torch.Tensor):
            The VAD decision of the audio file in the finest scale.
        vad_decision_base (torch.Tensor):
            The VAD decision of the audio file in the base scale.
    """
    rep_counts = torch.unique(scale_map[base_scale_idx], return_counts=True)[1]
    base_seg_inds = torch.cumsum(rep_counts, dim=0) - 1 # Pick the last index of each repeating index

    ms_silsp_embs = embeddings[:, :(base_scale_idx+1), :][base_seg_inds, :, :] # [T, num_scales, emb_dim, (num_of_channels)]
    ms_ts_scaled = time_stamps[base_scale_idx][base_seg_inds]/feat_per_sec
    vad_index_list = [0] + (base_seg_inds + 1).tolist() # selected scale's T + 1
    
    vad_prob_mat_list = []
    for i in range(len(vad_index_list)-1):
        vad_prob_mat_list.append(torch.mean(vad_probs[vad_index_list[i]:vad_index_list[i+1]]))
        
    vad_prob_mat = torch.stack(vad_prob_mat_list, dim=0)
    vad_prob_mat_base = vad_probs 
        
    hist_ct, bins = torch.histogram(vad_prob_mat, bins=50, range=(0, 1))
    hist_ct_norm = hist_ct / hist_ct.sum()
    vad_thres_knee_argmax = torch.argmax(hist_ct_norm[0:10] - hist_ct_norm[1:11])
    vad_thres_offset = bins[vad_thres_knee_argmax+1].item()
    vad_threshold = vad_thres_offset + vad_threshold
    logging.info(f"[VAD Thresholding] Adaptive || vad_threshold || is set to: [{vad_threshold:.3f}]")
    vad_decision_scaled = vad_prob_mat > vad_threshold
    vad_decision_base = vad_prob_mat_base > vad_threshold
    ms_embs_scaled_vadmasked = ms_silsp_embs[vad_decision_scaled, : , :]
    return ms_silsp_embs, ms_embs_scaled_vadmasked, ms_ts_scaled, vad_decision_scaled, vad_decision_base

def get_scaled_drop_length_thres(
    drop_length_thres: int,
    base_scale_idx: int, 
    clustering_scale_index: int, 
    multiscale_dict: Dict[int, Tuple[int, int]],
    ):
    """
    Get scaled drop length threshold.

    Args:
        drop_length_thres (int):
            The drop length threshold.
        base_scale_idx (int):
            The base scale index which is the highest scale index.
        clustering_scale_index (int):
            The clustering scale index.
        multiscale_dict (dict):
            The multi-scale dictionary.

    Returns:
        (int):
            The scaled drop length threshold.
    """
    return int((multiscale_dict[clustering_scale_index][0]/multiscale_dict[base_scale_idx][0]) * drop_length_thres)

def get_selected_channel_embs(
    ms_emb_seq, 
    max_mc_ch_num, 
    collapse_scale_dim: bool =False,
    multiscale_weights: list =[], 
    ):
    """
    Get selected channel embeddings for multi-channel speaker diarization.

    Args:
        ms_emb_seq (torch.Tensor):
            The multi-scale embeddings of the audio file.
        max_mc_ch_num (int):
            The maximum number of multi-channel embeddings.
        collapse_scale_dim (bool, optional):
            Whether to collapse the scale dimension. Defaults to False.
        multiscale_weights (list, optional):
            The multi-scale weights. Defaults to [].

    Returns:
        (torch.Tensor):
            The selected channel embeddings.
    """
    if collapse_scale_dim:
        if len(multiscale_weights) == 0: # If no weights are given, use equal weights
            multiscale_weights = [1.0 for _ in range(ms_emb_seq.shape[1])]
        multiscale_weights_tensor = torch.tensor(multiscale_weights).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        ms_emb_seq_weighted = ms_emb_seq * multiscale_weights_tensor[:, :ms_emb_seq.shape[1]]
        merged_mono_scale_embs = ms_emb_seq_weighted.sum(dim=1)
    else:
        merged_mono_scale_embs = ms_emb_seq.reshape(ms_emb_seq.shape[0], -1, ms_emb_seq.shape[-1]) # [T, scale_n, emb_dim, ch] -> [T, scale_n * emb_dim, ch]
    t_embs  = merged_mono_scale_embs.transpose(1, 2) # [T, ch, emb_dim]
    ch_sim = cos_similarity_batch(emb_a=t_embs.float(), emb_b=t_embs.float())  # [T, ch, ch]
    ch_sim_T = ch_sim.mean(dim=1)
    arg_sort_inds = torch.sort(ch_sim_T, descending=True)[1]
    if arg_sort_inds.shape[1] > max_mc_ch_num:
        arg_sort_inds = arg_sort_inds[:, :max_mc_ch_num]
    sorted_ch_inds = torch.sort(arg_sort_inds, dim=1, descending=True)[0]
    merged_mono_scale_embs_list = []
    for tdx in range(sorted_ch_inds.shape[0]):
        merged_mono_scale_embs_list.append(merged_mono_scale_embs[tdx, :, sorted_ch_inds[tdx,:]])
    if collapse_scale_dim:
        selected_ss_mc_embs = torch.stack(merged_mono_scale_embs_list, dim=0)
    else:
        ms_cat_emb_seq = torch.stack(merged_mono_scale_embs_list, dim=0)
        selected_ss_mc_embs = ms_cat_emb_seq.reshape(ms_emb_seq.shape[0], ms_emb_seq.shape[1], ms_emb_seq.shape[2], ms_cat_emb_seq.shape[-1])
    return selected_ss_mc_embs

def perform_clustering_embs(
    embeddings_dict: Dict[str, torch.Tensor],
    time_stamps_dict: Dict[str, torch.Tensor],
    vad_probs_dict: Dict[str, torch.Tensor],
    scale_mapping_dict: Dict[str, torch.Tensor],
    AUDIO_RTTM_MAP: dict,
    out_rttm_dir: str,
    clustering_params: omegaconf.DictConfig,
    multiscale_weights: list,
    device: torch.device,
    vad_threshold: float,
    multiscale_dict: dict, 
    verbose: bool = True,
    drop_length_thres = 4800,
    feat_per_sec: int = 100,
    long_audio_thres: int = 100000,
    unit_clus_len: int = 1000,
    get_rttm_with_the_finest_scale: bool = True,
):
    if len(embeddings_dict) == 0:
        raise ValueError("Empty embeddings_dict.")
    if len(time_stamps_dict) == 0:
        raise ValueError("Empty time_stamps_dict.")
    uniq_clus_labels_dict = {}
    cuda = True
    all_hypothesis, all_reference = [], []
    no_references = False
    lines_cluster_labels = []
    base_scale_idx = clustering_params.clustering_scale_index
    max_mc_ch_num = clustering_params.max_mc_ch_num 
    if device.type != 'cuda':
        if verbose:
            logging.warning("cuda=False, using CPU for eigen decomposition. This might slow down the clustering process.")
        cuda = False
    speaker_clustering = SpeakerClustering(cuda=cuda)
    # If True, export torch script module and save it to the base folder.
    for uniq_id, audio_rttm_values in tqdm(AUDIO_RTTM_MAP.items(), desc='clustering', leave=True, disable=not verbose):
        scale_map = scale_mapping_dict[uniq_id]
            
        if len(embeddings_dict[uniq_id].shape) > 3:
            # The last dimension is the channel dimension
            embeddings = embeddings_dict[uniq_id]
            time_stamps = time_stamps_dict[uniq_id][:, :, :, 0]
        else:
            embeddings = embeddings_dict[uniq_id]
            time_stamps = time_stamps_dict[uniq_id]
        
        vad_probs = vad_probs_dict[uniq_id]
        if scale_map.shape[1] > long_audio_thres:
            if verbose:
                logging.info(f"[Speaker Clustering] Long form audio detected: Using {base_scale_idx}-index scale length {multiscale_dict[base_scale_idx]} Segment Count - {scale_map.shape[1]}")
            base_scale_idx = max(0, base_scale_idx - 1)
            use_fine_grained_clustering = False
        else:
            if verbose:
                logging.info(f"[Speaker Clustering] Short form audio detected: Segment Count - {scale_map.shape[1]}")
            use_fine_grained_clustering = False
        
        ms_silsp_embs, ms_embs_scaled_vadmasked, ms_ts_scaled, vad_decision_scaled, vad_decision_base = get_ms_embs_and_ts(base_scale_idx, 
                                                                                                                           embeddings, 
                                                                                                                           time_stamps, 
                                                                                                                           scale_map, 
                                                                                                                           vad_probs, 
                                                                                                                           vad_threshold,
                                                                                                                           feat_per_sec)
        if len(ms_embs_scaled_vadmasked.shape) > 3: # This is multi-channel case
            selected_ss_mc_embs = get_selected_channel_embs(
                ms_embs_scaled_vadmasked, 
                max_mc_ch_num, 
                collapse_scale_dim=True,
                multiscale_weights=multiscale_weights, 
                )
        else:
            multiscale_weights_tensor = torch.tensor(multiscale_weights).float().unsqueeze(0).unsqueeze(-1)
            selected_ss_mc_embs = (ms_embs_scaled_vadmasked * multiscale_weights_tensor[:, :ms_embs_scaled_vadmasked.shape[1]]).sum(dim=1)
        
        if clustering_params.oracle_num_speakers:
            num_speakers = audio_rttm_values.get('num_speakers', None)
            if num_speakers is None:
                raise ValueError("Provided option as oracle num of speakers but num_speakers in manifest is null")
        else:
            num_speakers = -1
            
        drop_length_thres_scaled = get_scaled_drop_length_thres(drop_length_thres, 
                                                                base_scale_idx, 
                                                                clustering_params.clustering_scale_index, 
                                                                multiscale_dict)
        
        cluster_labels = speaker_clustering.forward_embs(
                embs=selected_ss_mc_embs,
                oracle_num_speakers=int(num_speakers),
                max_num_speakers=int(clustering_params.max_num_speakers),
                min_num_speakers=int(clustering_params.get('min_num_speakers', 1)),
                max_rp_threshold=float(clustering_params.max_rp_threshold),
                sparse_search_volume=int(clustering_params.sparse_search_volume),
                drop_length_thres=drop_length_thres_scaled,
            )
        
        cluster_labels_infer, max_scm = get_cluster_labels_infer(ms_silsp_embs, 
                                                                 cluster_labels, 
                                                                 vad_decision_scaled, 
                                                                 vad_decision_base, 
                                                                 scale_map, 
                                                                 base_scale_idx)
            
        # Mask non-speech again with the finest scale resolution.
        if use_fine_grained_clustering:
            logging.info(f"Using fine grained clustering on embedding shape : {embeddings_dict[uniq_id].shape} and unit_clus_len: {unit_clus_len}")
            cluster_base_labels = divide_and_conquer_clustering(embeddings, 
                                                                cluster_labels_infer, 
                                                                unit_clus_len=unit_clus_len, 
                                                                max_num_speakers=int(clustering_params.max_num_speakers),
                                                                base_scale_idx=base_scale_idx, 
                                                                sync_score_thres=clustering_params.sync_score_thres)
            cluster_labels_infer = cluster_base_labels.cpu()
        
        if cuda:
            torch.cuda.empty_cache()
        else:
            gc.collect()

        uniq_clus_labels_dict[uniq_id] = cluster_labels_infer
       
        del ms_embs_scaled_vadmasked, ms_silsp_embs, selected_ss_mc_embs, ms_ts_scaled, vad_decision_scaled, vad_decision_base
        if get_rttm_with_the_finest_scale: 
            timestamps = time_stamps[-1][:max_scm][cluster_labels_infer != -1]/feat_per_sec
            cluster_labels = cluster_labels_infer[cluster_labels_infer != -1].cpu().numpy()
        else:
            timestamps = ms_ts_scaled[vad_decision_scaled, :] 
            cluster_labels = cluster_labels.cpu().numpy()
        
        if len(cluster_labels) != timestamps.shape[0]:
            raise ValueError("Mismatch of length between cluster_labels and timestamps.")
        labels, lines = generate_cluster_labels(timestamps, cluster_labels)
        if out_rttm_dir:
            labels_to_rttmfile(labels, uniq_id, out_rttm_dir)
            lines_cluster_labels.extend([f'{uniq_id} {seg_line}\n' for seg_line in lines])
        hypothesis = labels_to_pyannote_object(labels, uniq_name=uniq_id)
        all_hypothesis.append([uniq_id, hypothesis])
        rttm_file = audio_rttm_values.get('rttm_filepath', None)
        if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
            ref_labels = rttm_to_labels(rttm_file)
            # ref_labels = get_partial_ref_labels(pred_labels=labels, ref_labels=ref_labels)
            reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
            all_reference.append([uniq_id, reference])
        else:
            no_references = True
            all_reference = []

    return all_reference, all_hypothesis, uniq_clus_labels_dict

def perform_clustering(
    embs_and_timestamps, AUDIO_RTTM_MAP, out_rttm_dir, clustering_params, device, verbose: bool = True
):
    """
    Performs spectral clustering on embeddings with time stamps generated from VAD output

    Args:
        embs_and_timestamps (dict): This dictionary contains the following items indexed by unique IDs.
            'embeddings' : Tensor containing embeddings. Dimensions:(# of embs) x (emb. dimension)
            'timestamps' : Tensor containing ime stamps list for each audio recording
            'multiscale_segment_counts' : Tensor containing the number of segments for each scale
        AUDIO_RTTM_MAP (dict): AUDIO_RTTM_MAP for mapping unique id with audio file path and rttm path
        out_rttm_dir (str): Path to write predicted rttms
        clustering_params (dict): clustering parameters provided through config that contains max_num_speakers (int),
        oracle_num_speakers (bool), max_rp_threshold(float), sparse_search_volume(int) and enhance_count_threshold (int)
        use_torch_script (bool): Boolean that determines whether to use torch.jit.script for speaker clustering
        device (torch.device): Device we are running on ('cpu', 'cuda').
        verbose (bool): Enable TQDM progress bar.

    Returns:
        all_reference (list[uniq_name,Annotation]): reference annotations for score calculation
        all_hypothesis (list[uniq_name,Annotation]): hypothesis annotations for score calculation

    """
    all_hypothesis = []
    all_reference = []
    no_references = False
    lines_cluster_labels = []

    cuda = True
    if device.type != 'cuda':
        logging.warning("cuda=False, using CPU for eigen decomposition. This might slow down the clustering process.")
        cuda = False

    speaker_clustering = LongFormSpeakerClustering(cuda=cuda)

    if clustering_params.get('export_script_module', False):
        speaker_clustering = torch.jit.script(speaker_clustering)
        torch.jit.save(speaker_clustering, 'speaker_clustering_script.pt')

    for uniq_id, audio_rttm_values in tqdm(AUDIO_RTTM_MAP.items(), desc='clustering', leave=True, disable=not verbose):
        uniq_embs_and_timestamps = embs_and_timestamps[uniq_id]

        if clustering_params.oracle_num_speakers:
            num_speakers = audio_rttm_values.get('num_speakers', None)
            if num_speakers is None:
                raise ValueError("Provided option as oracle num of speakers but num_speakers in manifest is null")
        else:
            num_speakers = -1

        base_scale_idx = uniq_embs_and_timestamps['multiscale_segment_counts'].shape[0] - 1
        
        cluster_labels = speaker_clustering.forward_infer(
            embeddings_in_scales=uniq_embs_and_timestamps['embeddings'],
            timestamps_in_scales=uniq_embs_and_timestamps['timestamps'],
            multiscale_segment_counts=uniq_embs_and_timestamps['multiscale_segment_counts'],
            multiscale_weights=uniq_embs_and_timestamps['multiscale_weights'],
            oracle_num_speakers=int(num_speakers),
            max_num_speakers=int(clustering_params.max_num_speakers),
            min_num_speakers=int(clustering_params.get('min_num_speakers', 1)),
            max_rp_threshold=float(clustering_params.max_rp_threshold),
            sparse_search_volume=int(clustering_params.sparse_search_volume),
            chunk_cluster_count=clustering_params.get('chunk_cluster_count', None),
            embeddings_per_chunk=clustering_params.get('embeddings_per_chunk', None),
        )

        del uniq_embs_and_timestamps
        if cuda:
            torch.cuda.empty_cache()
        else:
            gc.collect()
        timestamps = speaker_clustering.timestamps_in_scales[base_scale_idx]

        cluster_labels = cluster_labels.cpu().numpy()
        if len(cluster_labels) != timestamps.shape[0]:
            raise ValueError("Mismatch of length between cluster_labels and timestamps.")

        labels, lines = generate_cluster_labels(timestamps, cluster_labels)

        if out_rttm_dir:
            labels_to_rttmfile(labels, uniq_id, out_rttm_dir)
            lines_cluster_labels.extend([f'{uniq_id} {seg_line}\n' for seg_line in lines])
        hypothesis = labels_to_pyannote_object(labels, uniq_name=uniq_id)
        all_hypothesis.append([uniq_id, hypothesis])
        

        rttm_file = audio_rttm_values.get('rttm_filepath', None)
        if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
            ref_labels = rttm_to_labels(rttm_file)
            reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
            all_reference.append([uniq_id, reference])
        else:
            no_references = True
            all_reference = []

    if out_rttm_dir:
        write_cluster_labels(base_scale_idx, lines_cluster_labels, out_rttm_dir)

    return all_reference, all_hypothesis


def get_vad_out_from_rttm_line(rttm_line):
    """
    Extract VAD timestamp from the given RTTM lines.
    """
    vad_out = rttm_line.strip().split()
    if len(vad_out) > 3:
        start, dur, _ = float(vad_out[3]), float(vad_out[4]), vad_out[7]
    else:
        start, dur, _ = float(vad_out[0]), float(vad_out[1]), vad_out[2]
    return start, dur


def get_offset_and_duration(AUDIO_RTTM_MAP, uniq_id, decimals=5):
    """
    Extract offset and duration information from AUDIO_RTTM_MAP dictionary.
    If duration information is not specified, a duration value is extracted from the audio file directly.

    Args:
        AUDIO_RTTM_MAP (dict):
            Dictionary containing RTTM file information, which is indexed by unique file id.
        uniq_id (str):
            Unique file id
    Returns:
        offset (float):
            The offset value that determines the beginning of the audio stream.
        duration (float):
            The length of audio stream that is expected to be used.
    """
    audio_path = AUDIO_RTTM_MAP[uniq_id]['audio_filepath']
    if AUDIO_RTTM_MAP[uniq_id].get('duration', None):
        duration = round(AUDIO_RTTM_MAP[uniq_id]['duration'], decimals)
        offset = round(AUDIO_RTTM_MAP[uniq_id]['offset'], decimals)
    else:
        sound = sf.SoundFile(audio_path)
        duration = sound.frames / sound.samplerate
        offset = 0.0
    return offset, duration


def write_overlap_segments(outfile, AUDIO_RTTM_MAP, uniq_id, overlap_range_list, decimals=5):
    """
    Write the json dictionary into the specified manifest file.

    Args:
        outfile:
            File pointer that indicates output file path.
        AUDIO_RTTM_MAP (dict):
            Dictionary containing the input manifest information
        uniq_id (str):
            Unique file id
        overlap_range_list (list):
            List containing overlapping ranges between target and source.
        decimals (int):
            Number of decimals to round the offset and duration values.
    """
    audio_path = AUDIO_RTTM_MAP[uniq_id]['audio_filepath']
    for (stt, end) in overlap_range_list:
        meta = {
            "audio_filepath": audio_path,
            "offset": round(stt, decimals),
            "duration": round(end - stt, decimals),
            "label": 'UNK',
            "uniq_id": uniq_id,
        }
        json.dump(meta, outfile)
        outfile.write("\n")

def write_diarized_segments(outfile_path, json_dict_list):
    """
    Write the json dictionary into the specified manifest file.
    """
    with open(outfile_path, 'w') as outfile:
        for meta in json_dict_list:
            json.dump(meta, outfile)
            outfile.write("\n")

def read_rttm_lines(rttm_file_path):
    """
    Read rttm files and return the rttm information lines.

    Args:
        rttm_file_path (str):
            An absolute path to an RTTM file

    Returns:
        lines (list):
            List containing the strings from the RTTM file.
    """
    if rttm_file_path and os.path.exists(rttm_file_path):
        with open(rttm_file_path, 'r') as f:
            lines = f.readlines()
    else:
        raise FileNotFoundError(
            "Requested to construct manifest from rttm with oracle VAD option or from NeMo VAD but received filename as {}".format(
                rttm_file_path
            )
        )
    return lines


def validate_vad_manifest(AUDIO_RTTM_MAP, vad_manifest):
    """
    This function will check the valid speech segments in the manifest file which is either
    generated from NeMo voice activity detection(VAD) or oracle VAD.
    If an audio file does not contain any valid speech segments, we ignore the audio file
    (indexed by uniq_id) for the rest of the processing steps.
    """
    vad_uniq_ids = set()
    with open(vad_manifest, 'r') as vad_file:
        for line in vad_file:
            line = line.strip()
            dic = json.loads(line)
            if dic['duration'] > 0:
                vad_uniq_ids.add(dic['uniq_id'])

    provided_uniq_ids = set(AUDIO_RTTM_MAP.keys())
    silence_ids = provided_uniq_ids - vad_uniq_ids
    for uniq_id in silence_ids:
        del AUDIO_RTTM_MAP[uniq_id]
        logging.warning(f"{uniq_id} is ignored since the file does not contain any speech signal to be processed.")

    if len(AUDIO_RTTM_MAP) == 0:
        raise ValueError("All files present in manifest contains silence, aborting next steps")


def is_overlap(rangeA: List[float], rangeB: List[float]) -> bool:
    """
    Check whether two ranges have overlap.

    Args:
        rangeA (list, tuple):
            List or tuple containing start and end value in float.
        rangeB (list, tuple):
            List or tuple containing start and end value in float.
    Returns:
        (bool):
            Boolean that indicates whether the input ranges have overlap.
    """
    start1, end1 = rangeA[0], rangeA[1]
    start2, end2 = rangeB[0], rangeB[1]
    return end1 > start2 and end2 > start1


def get_overlap_range(rangeA: List[float], rangeB: List[float]):
    """
    Calculate the overlapping range between rangeA and rangeB.

    Args:
        rangeA (list, tuple):
            List or tuple containing start and end value in float.
        rangeB (list, tuple):
            List or tuple containing start and end value in float.

    Returns:
        (list):
            List containing the overlapping range between rangeA and rangeB.
    """
    assert is_overlap(rangeA, rangeB), f"There is no overlap between rangeA:{rangeA} and rangeB:{rangeB}"
    return [max(rangeA[0], rangeB[0]), min(rangeA[1], rangeB[1])]


def merge_int_intervals(intervals_in: List[List[int]]) -> List[List[int]]:
    """
    Interval merging algorithm which has `O(N*logN)` time complexity. (N is number of intervals)
    Merge the range pairs if there is overlap exists between the given ranges.
    This algorithm needs a sorted range list in terms of the start time.
    Note that neighboring numbers lead to a merged range.

    Example:
        input: [(1, 10), (11, 20)]
        output: [(1, 20)]

    Refer to the original code at https://stackoverflow.com/a/59378428

    Args:
        intervals_in (list):
            List containing ranges.
            Example:
                >>> intervals_in
                [(102, 103), (104, 109), (107, 120)]

    Returns:
        merged_list (list):
            List containing the combined ranges.
            Example:
                >>> merged_list
                [(102, 120)]
    """
    num_intervals = len(intervals_in)
    if num_intervals == 0:
        return []
    elif num_intervals == 1:
        return intervals_in
    else:
        merged_list: List[List[int]] = []
        stt2: int = 0
        end2: int = 0

        intervals_in = [[int(x[0]), int(x[1])] for x in intervals_in]
        interval_tensor: torch.Tensor = torch.tensor(intervals_in)
        _sorted, _ = torch.sort(interval_tensor, dim=0)
        _sorted_int: List[List[int]] = [[int(x[0]), int(x[1])] for x in _sorted.cpu()]
        intervals: List[List[int]] = _sorted_int

        start, end = intervals[0][0], intervals[0][1]
        for i in range(1, num_intervals):
            stt2, end2 = intervals[i][0], intervals[i][1]
            if end >= stt2:
                end = max(end2, end)
            else:
                start, end = int(start), int(end)
                merged_list.append([start, end])
                start = stt2
                end = max(end2, end)

        start, end = int(start), int(end)
        merged_list.append([start, end])
        return merged_list


def fl2int(x: float, decimals: int = 3) -> int:
    """
    Convert floating point number to integer.
    """
    return torch.round(torch.tensor([x * (10 ** decimals)]), decimals=0).int().item()


def int2fl(x: int, decimals: int = 3) -> float:
    """
    Convert integer to floating point number.
    """
    return torch.round(torch.tensor([x / (10 ** decimals)]), decimals=decimals).item()


def merge_float_intervals(ranges: List[List[float]], decimals: int = 5, margin: int = 2) -> List[List[float]]:
    """
    Combine overlaps with floating point numbers. Since neighboring integers are considered as continuous range,
    we need to add margin to the starting range before merging then subtract margin from the result range.

    Args:
        ranges (list):
            List containing ranges.
            Example: [(10.2, 10.83), (10.42, 10.91), (10.45, 12.09)]
        decimals (int):
            Number of rounding decimals
        margin (int):
            margin for determining overlap of the two ranges when ranges are converted to integer ranges.
            Default is margin=2 which follows the python index convention.

        Examples:
            If margin is 0:
                [(1, 10), (10, 20)] -> [(1, 20)]
                [(1, 10), (11, 20)] -> [(1, 20)]
            If margin is 1:
                [(1, 10), (10, 20)] -> [(1, 20)]
                [(1, 10), (11, 20)] -> [(1, 10), (11, 20)]
            If margin is 2:
                [(1, 10), (10, 20)] -> [(1, 10), (10, 20)]
                [(1, 10), (11, 20)] -> [(1, 10), (11, 20)]

    Returns:
        merged_list (list):
            List containing the combined ranges.
            Example: [(10.2, 12.09)]
    """
    ranges_int: List[List[int]] = []
    merged_ranges_int: List[List[int]] = []
    for x in ranges:
        stt, end = int(fl2int(x[0], decimals) + margin), int(fl2int(x[1], decimals))
        if stt < end:
            ranges_int.append([stt, end])
    merged_ranges_int = merge_int_intervals(ranges_int)
    merged_ranges_float: List[List[float]] = []
    merged_ranges_float = [[int2fl(x[0] - margin, decimals), int2fl(x[1], decimals)] for x in merged_ranges_int]
    return merged_ranges_float


def get_sub_range_list(target_range: List[float], source_range_list: List[List[float]]) -> List[List[float]]:
    """
    Get the ranges that has overlaps with the target range from the source_range_list.

    Example:
        source range:
            |===--======---=====---====--|
        target range:
            |--------================----|
        out_range:
            |--------===---=====---==----|

    Args:
        target_range (list):
            A range (a start and end value pair) that defines the target range we want to select.
            target_range = [(start, end)]
        source_range_list (list):
            List containing the subranges that need to be selected.
            source_range = [(start0, end0), (start1, end1), ...]
    Returns:
        out_range (list):
            List containing the overlap between target_range and
            source_range_list.
    """
    if len(target_range) == 0:
        return []
    else:
        out_range: List[List[float]] = []
        for s_range in source_range_list:
            if is_overlap(s_range, target_range):
                ovl_range = get_overlap_range(s_range, target_range)
                out_range.append(ovl_range)
        return out_range


def write_rttm2manifest(
    AUDIO_RTTM_MAP: str, manifest_file: str, include_uniq_id: bool = False, decimals: int = 5
) -> str:
    """
    Write manifest file based on rttm files (or vad table out files). This manifest file would be used by
    speaker diarizer to compute embeddings and cluster them. This function takes care of overlapping VAD timestamps
    and trimmed with the given offset and duration value.

    Args:
        AUDIO_RTTM_MAP (dict):
            Dictionary containing keys to unique names, that contains audio filepath and rttm_filepath as its contents,
            these are used to extract oracle vad timestamps.
        manifest (str):
            The path to the output manifest file.

    Returns:
        manifest (str):
            The path to the output manifest file.
    """
    with open(manifest_file, 'w') as outfile:
        for uniq_id in AUDIO_RTTM_MAP:
            rttm_file_path = AUDIO_RTTM_MAP[uniq_id]['rttm_filepath']
            rttm_lines = read_rttm_lines(rttm_file_path)
            offset, duration = get_offset_and_duration(AUDIO_RTTM_MAP, uniq_id, decimals)
            vad_start_end_list_raw = []
            for line in rttm_lines:
                start, dur = get_vad_out_from_rttm_line(line)
                vad_start_end_list_raw.append([start, start + dur])
            vad_start_end_list = merge_float_intervals(vad_start_end_list_raw, decimals)
            if len(vad_start_end_list) == 0:
                logging.warning(f"File ID: {uniq_id}: The VAD label is not containing any speech segments.")
            elif duration <= 0:
                logging.warning(f"File ID: {uniq_id}: The audio file has negative or zero duration.")
            else:
                overlap_range_list = get_sub_range_list(
                    source_range_list=vad_start_end_list, target_range=[offset, offset + duration]
                )
                write_overlap_segments(outfile, AUDIO_RTTM_MAP, uniq_id, overlap_range_list, decimals)
    return manifest_file


def segments_manifest_to_subsegments_manifest(
    segments_manifest_file: str,
    subsegments_manifest_file: str = None,
    window: float = 1.5,
    shift: float = 0.75,
    min_subsegment_duration: float = 0.05,
    include_uniq_id: bool = False,
):
    """
    Generate subsegments manifest from segments manifest file
    Args:
        segments_manifest file (str): path to segments manifest file, typically from VAD output
        subsegments_manifest_file (str): path to output subsegments manifest file (default (None) : writes to current working directory)
        window (float): window length for segments to subsegments length
        shift (float): hop length for subsegments shift
        min_subsegments_duration (float): exclude subsegments smaller than this duration value

    Returns:
        returns path to subsegment manifest file
    """
    if subsegments_manifest_file is None:
        pwd = os.getcwd()
        subsegments_manifest_file = os.path.join(pwd, 'subsegments.json')

    with open(segments_manifest_file, 'r') as segments_manifest, open(
        subsegments_manifest_file, 'w'
    ) as subsegments_manifest:
        segments = segments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            if include_uniq_id and 'uniq_id' in dic and dic['uniq_id'] is not None:
                uniq_id = dic['uniq_id']
            else:
                uniq_id = None
            for subsegment in subsegments:
                start, dur = subsegment
                if dur > min_subsegment_duration:
                    meta = {
                        "audio_filepath": audio,
                        "offset": start,
                        "duration": dur,
                        "label": label,
                        "uniq_id": uniq_id,
                    }

                    json.dump(meta, subsegments_manifest)
                    subsegments_manifest.write("\n")

    return subsegments_manifest_file

def get_subsegments(
    offset: float, 
    window: float, 
    shift: float, 
    duration: float, 
    min_subsegment_duration: float = 0.03,
    decimals: int = 2,
    ) -> List[List[float]]:
    """
    Return subsegments from a segment of audio file.
    
    Example:
        (window, shift) = 1.5, 0.75
        Segment:  [12.05, 14.45]    
        Subsegments: [[12.05, 13.55], [12.8, 14.3], [13.55, 14.45], [14.3, 14.45]]

    Args:
        offset (float): start time of audio segment
        window (float): window length for segments to subsegments length
        shift (float): hop length for subsegments shift
        duration (float): duration of segment
    Returns:
        subsegments (List[tuple[float, float]]): subsegments generated for the segments as list of tuple of start and duration of each subsegment
    """
    subsegments:  List[List[float]] = []
    start = offset
    slice_end = start + duration
    # base = math.ceil((duration - window) / shift)
    # slices = 1 if base < 0 else base + 1
    if min_subsegment_duration <= duration < shift:
        slices = 1
    else:
        slices = int(np.ceil((duration-window)/shift) + 1)
    if slices == 1:
        if min(duration, window) >= min_subsegment_duration:
            subsegments.append([start, min(duration, window)])
    else:
        start_col = torch.arange(offset, slice_end, shift)[:slices]
        dur_col = window * torch.ones(slices)
        dur_col[-1] = min(slice_end - start_col[-1], window)
        dur_col = torch.round(dur_col, decimals=decimals)
        ss_tensor = torch.stack([start_col, dur_col], dim=1)
        for k in range(ss_tensor.shape[0]):
            if dur_col[k] >= min_subsegment_duration:
                subsegments.append([float(ss_tensor[k,0].item()), float(ss_tensor[k,1].item())])
    return subsegments

def get_subsegments_(offset: float, window: float, shift: float, duration: float) -> List[List[float]]:
    """
    Return subsegments from a segment of audio file
    Args:
        offset (float): start time of audio segment
        window (float): window length for segments to subsegments length
        shift (float): hop length for subsegments shift
        duration (float): duration of segment
    Returns:
        subsegments (List[tuple[float, float]]): subsegments generated for the segments as list of tuple of start and duration of each subsegment
    """
    subsegments: List[List[float]] = []
    start = offset
    slice_end = start + duration
    base = math.ceil((duration - window) / shift)
    slices = 1 if base < 0 else base + 1
    for slice_id in range(slices):
        end = start + window
        if end > slice_end:
            end = slice_end
        subsegments.append([start, end - start])
        start = offset + (slice_id + 1) * shift
    return subsegments


def get_target_sig(sig, start_sec: float, end_sec: float, slice_length: int, sample_rate: int,) -> torch.Tensor:
    """
    Extract time-series signal from the given audio buffer based on the start and end
    timestamps.

    Args:
        start_sec (float):
            Start of the targeted segments in second
        end_sec (float):
            Start of the targeted segments in second
        slice_length (int):
            Length of the entire audio segment that the samples are extracted from
        sample_rate (int):
            Sampling rate of the time-series audio signal

    Returns:
        (Tensor) Trimmed ime-series audio signal samples
    """
    start_idx = int(start_sec * sample_rate)
    end_idx = min(int(end_sec * sample_rate), int(slice_length + start_idx))
    return sig[start_idx:end_idx]


def check_ranges(range_tensor):
    """
    Check whether the range list has any faulty timestamp order.

    Args:
        range_tensor (list):
            List containing the start and end time of the segments.
            Example:
                >>> range_tensor = [[0.5, 3.12], [3.51, 7.26], ... ]
    """
    for k in range(range_tensor.shape[0]):
        range_tup = range_tensor[k]
        if range_tup[1] < range_tup[0]:
            raise ValueError("Range start time should be preceding the end time but we got: {range_tup}")
    return True


def tensor_to_list(range_tensor: torch.Tensor) -> List[List[float]]:
    """
    For online segmentation. Force the list elements to be float type.
    """
    return [[float(range_tensor[k][0]), float(range_tensor[k][1])] for k in range(range_tensor.shape[0])]


def get_speech_labels_for_update(
    frame_start: float,
    buffer_end: float,
    vad_timestamps: torch.Tensor,
    cumulative_speech_labels: torch.Tensor,
    cursor_for_old_segments: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bring the new speech labels from the current buffer. Followingly:

    1. Concatenate the old speech labels from self.cumulative_speech_labels for the overlapped region.
        - This goes to new_speech_labels.
    2. Update the new 1 sec of speech label (speech_label_for_new_segments) to self.cumulative_speech_labels.
    3. Return the speech label from cursor_for_old_segments to buffer end.

    Args:
        frame_start (float):
            Start of the middle audio chunk in the audio buffer
        buffer_end (float):
            End of the audio buffer
        vad_timestamps (Tensor):
            Tensor containing VAD intervals (start and end timestamps)
        cumulative_speech_labels (torch.Tensor):
            Cumulative speech/non-speech timestamps (equivalent to VAD timestamps)
        cursor_for_old_segments (float):
            Floating point number that indicates the point where new segments should replace
            the old segments

    Returns:
        speech_label_for_new_segments (Tensor):
            The intervals (start and end) timestamps where the new incoming speech segments should
            be collected from
        cumulative_speech_labels (Tensor):
            Cumulative speech/non-speech timestamps (equivalent to VAD timestamps) with newly added
            speech/non-speech timestamps from the `vad_timestamps` input
    """
    update_overlap_range: List[float] = []
    if cursor_for_old_segments < frame_start:
        update_overlap_range = [float(cursor_for_old_segments), float(frame_start)]

    # Get VAD timestamps that are in (frame_start, buffer_end) range
    vad_timestamps = tensor_to_list(vad_timestamps)
    cumulative_speech_labels = tensor_to_list(cumulative_speech_labels)
    new_incoming_speech_labels = get_sub_range_list(
        target_range=[float(frame_start), float(buffer_end)], source_range_list=vad_timestamps
    )

    # Update the speech label by including overlapping region with the previous output
    update_overlap_speech_labels = get_sub_range_list(
        target_range=update_overlap_range, source_range_list=cumulative_speech_labels
    )

    # Speech segments for embedding extractions
    speech_label_for_new_segments = merge_float_intervals(
        update_overlap_speech_labels + new_incoming_speech_labels, margin=0
    )

    # Keep cumulative VAD labels for the future use
    cumulative_speech_labels = merge_float_intervals(cumulative_speech_labels + new_incoming_speech_labels, margin=0)

    # Convert the lists back to type torch.Tensor
    speech_label_for_new_segments = torch.tensor(speech_label_for_new_segments)
    cumulative_speech_labels = torch.tensor(cumulative_speech_labels)

    return speech_label_for_new_segments, cumulative_speech_labels


def get_new_cursor_for_update(frame_start: float, segment_range_ts: List[List[float]],) -> Tuple[float, int]:
    """
    Function for updating a cursor online speaker diarization. 
    Remove the old segments that overlap with the new frame (self.frame_start)
    cursor_for_old_segments is set to the onset of the t_range popped lastly.


    Args:
        frame_start (float):
            Start of streaming pipeline frame
        segment_range_ts (float):
            Interval (start and end timestamps) of the targeted segments

    Returns:
        cursor_for_old_segments (float):
            Floating point number that indicates the point where new segments should replace
            the old segments
        cursor_index (int):
            The index of the first newly accepted segments
    """
    cursor_for_old_segments = frame_start
    cursor_index: int = len(segment_range_ts)
    count = 0
    while True and len(segment_range_ts) > 0:
        t_range = segment_range_ts[-1 * (count + 1)]
        if frame_start <= t_range[1]:
            count += 1
            cursor_for_old_segments = t_range[0]
        else:
            break
    cursor_index = len(segment_range_ts) - count
    return cursor_for_old_segments, cursor_index


def get_online_segments_from_slices(
    sig: torch.Tensor,
    buffer_start: float,
    buffer_end: float,
    subsegments: List[List[float]],
    ind_offset: int,
    window: float,
    sample_rate: int,
) -> Tuple[int, List[torch.Tensor], List[List[float]], List[int]]:
    """
    Create short speech segments from slices for online processing purpose.

    Args:
        sig (Tensor):
            Tensor containing the raw time-series signal
        buffer_start (float):
            Start point of the time-series signal buffer
        buffer_end (float):
            End point of the time-series signal buffer
        subsegments (list):
            List containing the interval information (start and duration) of each segment
        ind_offset (int):
            Offset for index that compensates the point of the current position in the streaming session
        window (float):
            Window length in second
        shift (float):
            Shift length in second

    Returns:
        sigs_list  (list):
            list of sliced input signal
        audio_lengths (list):
            list of audio sample lengths
    """
    sig_rangel_list: List[List[float]] = []
    sig_indexes: List[int] = []
    sigs_list: List[torch.Tensor] = []
    slice_length: int = int(window * sample_rate)
    end_sec: float = 0.0
    for subseg in subsegments:
        start_sec, dur = subseg[0], subseg[1]

        if start_sec > buffer_end:
            continue
        ind_offset += 1

        buffer_len = buffer_end - buffer_start
        end_sec = float(start_sec + dur)

        if end_sec > buffer_len:
            end_sec = float(min(end_sec, buffer_len))

        signal = get_target_sig(sig, start_sec, end_sec, slice_length, sample_rate)

        if len(signal) == 0:
            raise ValueError("len(signal) is zero. Signal length should not be zero.")
        if len(signal) < slice_length:
            signal = repeat_signal(signal, len(signal), slice_length)

        start_abs_sec = buffer_start + start_sec
        end_abs_sec = buffer_start + end_sec

        sigs_list.append(signal)
        sig_rangel_list.append([start_abs_sec, end_abs_sec])
        sig_indexes.append(ind_offset)

    if not len(sigs_list) == len(sig_rangel_list) == len(sig_indexes):
        raise ValueError("Signal information lists have a mismatch.")

    return ind_offset, sigs_list, sig_rangel_list, sig_indexes


def get_online_subsegments_from_buffer(
    buffer_start: float,
    buffer_end: float,
    sample_rate: int,
    speech_labels_for_update: torch.Tensor,
    audio_buffer: torch.Tensor,
    segment_indexes: List[int],
    window: float,
    shift: float,
) -> Tuple[List[torch.Tensor], List[List[float]], List[int]]:
    """
    Generate subsegments for online processing from the given segment information.
    This function extracts subsegments (embedding vector level) time-series from the
    raw time-series buffer based on the segment interval (start and end timestamps) information.

    Args:
        buffer_start (float):
            Start point of the time-series signal buffer
        buffer_end (float):
            End point of the time-series signal buffer
        sample_rate (int):
            Sampling rate of the audio input
        speech_labels_for_update (Tensor):
            Tensor containing intervals (start and end timestamps) of the speech segments
        audio_buffer (Tensor):
            Tensor containing the raw time-series signal
        segment_indexes (list):
            List containing the unique indices of segments
        window (float):
            Window length in second
        shift (float):
            Shift length in second

    Returns:
        sigs_list (list):
            List containing the tensors of the old and the newly added time-series signals
        sig_rangel_list (list):
            List containing the old and the newly added intervals (timestamps) of the speech segments
        sig_indexes (list):
            List containing the old and the newly added unique indices of segments
    """
    sigs_list: List[torch.Tensor] = []
    sig_rangel_list: List[List[float]] = []
    sig_indexes: List[int] = []
    if len(segment_indexes) > 0:
        ind_offset = segment_indexes[-1]
    else:
        ind_offset = -1

    for idx, range_spl in enumerate(speech_labels_for_update):
        range_offs = [float(range_spl[0].item() - buffer_start), float(range_spl[1].item() - buffer_start)]
        range_t = [max(0, range_offs[0]), range_offs[1]]

        subsegments = get_subsegments(
            offset=range_t[0], window=window, shift=shift, duration=(range_t[1] - range_t[0]),
        )
        ind_offset, sigs, ranges, inds = get_online_segments_from_slices(
            sig=audio_buffer,
            buffer_start=buffer_start,
            buffer_end=buffer_end,
            subsegments=subsegments,
            window=window,
            ind_offset=ind_offset,
            sample_rate=sample_rate,
        )

        sigs_list.extend(sigs)
        sig_rangel_list.extend(ranges)
        sig_indexes.extend(inds)

    assert len(sigs_list) == len(sig_rangel_list) == len(sig_indexes)
    return sigs_list, sig_rangel_list, sig_indexes


def get_scale_mapping_argmat(uniq_embs_and_timestamps: Dict[str, dict]) -> Dict[int, torch.Tensor]:
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.

    Args:
        uniq_embs_and_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_embs_and_timestamps contains only one scale, single scale diarization
            is performed.

    Returns:
        scale_mapping_argmat (dict)
            Dictionary containing scale mapping information matrix for each scale.
    """
    scale_mapping_argmat = {}
    embeddings_in_scales, timestamps_in_scales = split_input_data(
        embeddings_in_scales=uniq_embs_and_timestamps['embeddings'],
        timestamps_in_scales=uniq_embs_and_timestamps['timestamps'],
        multiscale_segment_counts=uniq_embs_and_timestamps['multiscale_segment_counts'],
    )
    session_scale_mapping_list = get_argmin_mat(timestamps_in_scales)
    for scale_idx in range(len(session_scale_mapping_list)):
        mapping_argmat = session_scale_mapping_list[scale_idx]
        scale_mapping_argmat[scale_idx] = mapping_argmat
    return scale_mapping_argmat


def get_overlap_stamps(cont_stamps: List[str], ovl_spk_idx: List[str]):
    """
    Generate timestamps that include overlap speech. Overlap-including timestamps are created based on the segments that are
    created for clustering diarizer. Overlap speech is assigned to the existing speech segments in `cont_stamps`.

    Args:
        cont_stamps (list):
            Non-overlapping (single speaker per segment) diarization output in string format.
            Each line contains the start and end time of segments and corresponding speaker labels.
        ovl_spk_idx (list):
            List containing segment index of the estimated overlapped speech. The start and end of segments are based on the
            single-speaker (i.e., non-overlap-aware) RTTM generation.
    Returns:
        total_ovl_cont_list (list):
            Rendered diarization output in string format. Each line contains the start and end time of segments and
            corresponding speaker labels. This format is identical to `cont_stamps`.
    """
    ovl_spk_cont_list = [[] for _ in range(len(ovl_spk_idx))]
    for spk_idx in range(len(ovl_spk_idx)):
        for idx, cont_a_line in enumerate(cont_stamps):
            start, end, speaker = cont_a_line.split()
            if idx in ovl_spk_idx[spk_idx]:
                ovl_spk_cont_list[spk_idx].append(f"{start} {end} speaker_{spk_idx}")
    total_ovl_cont_list = []
    for ovl_cont_list in ovl_spk_cont_list:
        if len(ovl_cont_list) > 0:
            total_ovl_cont_list.extend(merge_stamps(ovl_cont_list))
    return total_ovl_cont_list


def get_adaptive_threshold(estimated_num_of_spks: int, min_threshold: float, overlap_infer_spk_limit: int):
    """
    This function controls the magnitude of the sigmoid threshold based on the estimated number of speakers. As the number of
    speakers becomes larger, diarization error rate is very sensitive on overlap speech detection. This function linearly increases
    the threshold in proportion to the estimated number of speakers so more confident overlap speech results are reflected when
    the number of estimated speakers are relatively high.

    Args:
        estimated_num_of_spks (int):
            Estimated number of speakers from the clustering result.
        min_threshold (float):
            Sigmoid threshold value from the config file. This threshold value is minimum threshold value when `estimated_num_of_spks=2`
        overlap_infer_spk_limit (int):
            If the `estimated_num_of_spks` is less then `overlap_infer_spk_limit`, overlap speech estimation is skipped.

    Returns:
        adaptive_threshold (float):
            Threshold value that is scaled based on the `estimated_num_of_spks`.
    """
    adaptive_threshold = min_threshold - (estimated_num_of_spks - 2) * (min_threshold - 1) / (
        overlap_infer_spk_limit - 2
    )
    return adaptive_threshold

def generate_speaker_timestamps_(
    clus_labels: List[Union[float, int]], 
    msdd_preds: List[torch.Tensor], 
    timestamps, 
    **params
) -> Tuple[List[str], List[str]]:
    '''
    Generate speaker timestamps from the segmentation information. If `use_clus_as_main=True`, use clustering result for main speaker
    labels and use timestamps from the predicted sigmoid values. In this function, the main speaker labels in `maj_labels` exist for
    every subsegment steps while overlap speaker labels in `ovl_labels` only exist for segments where overlap-speech is occuring.

    Args:
        clus_labels (list):
            List containing integer-valued speaker clustering results.
        msdd_preds (list):
            List containing tensors of the predicted sigmoid values.
            Each tensor has shape of: (Session length, estimated number of speakers).
        params:
            Parameters for generating RTTM output and evaluation. Parameters include:
                infer_overlap (bool): If False, overlap-speech will not be detected.
                use_clus_as_main (bool): Add overlap-speech detection from MSDD to clustering results. If False, only MSDD output
                                         is used for constructing output RTTM files.
                overlap_infer_spk_limit (int): Above this limit, overlap-speech detection is bypassed.
                use_adaptive_thres (bool): Boolean that determines whehther to use adaptive_threshold depending on the estimated
                                           number of speakers.
                max_overlap_spks (int): Maximum number of overlap speakers detected. Default is 2.
                threshold (float): Sigmoid threshold for MSDD output.

    Returns:
        maj_labels (list):
            List containing string-formated single-speaker speech segment timestamps and corresponding speaker labels.
            Example: [..., '551.685 552.77 speaker_1', '552.99 554.43 speaker_0', '554.97 558.19 speaker_0', ...]
        ovl_labels (list):
            List containing string-formated additional overlapping speech segment timestamps and corresponding speaker labels.
            Note that `ovl_labels` includes only overlapping speech that is not included in `maj_labels`.
            Example: [..., '152.495 152.745 speaker_1', '372.71 373.085 speaker_0', '554.97 555.885 speaker_1', ...]
    '''
    if torch.isnan(msdd_preds).any():
        raise ValueError("MSDD output `msdd_preds` contains NaN values. Please check the input data.")
    msdd_preds.squeeze(0)
    estimated_num_of_spks = msdd_preds.shape[-1]
    overlap_speaker_list = [[] for _ in range(estimated_num_of_spks)]
    infer_overlap = estimated_num_of_spks < int(params['overlap_infer_spk_limit'])
    main_speaker_lines = []

    params['use_clus_as_main'] = True
    infer_overlap = False
    threshold = params['threshold']
    for seg_idx, cluster_label in enumerate(clus_labels):
        msdd_preds.squeeze(0)
        spk_for_seg = (msdd_preds[seg_idx] > threshold).int().cpu().numpy().tolist()
        sm_for_seg = msdd_preds[seg_idx].cpu().numpy()

        if params['use_clus_as_main']:
            main_spk_idx = int(cluster_label)
        else:
            main_spk_idx = np.argsort(msdd_preds[seg_idx].cpu().numpy())[::-1][0]
        if sum(spk_for_seg) > 1 and infer_overlap:
            idx_arr = np.argsort(sm_for_seg)[::-1]
            for ovl_spk_idx in idx_arr[: params['max_overlap_spks']].tolist():
                if ovl_spk_idx != int(main_spk_idx):
                    overlap_speaker_list[ovl_spk_idx].append(seg_idx)
        if params['use_clus_as_main']:
            main_spk_idx = int(cluster_label)
            main_speaker_lines.append(f"{timestamps[seg_idx][0]} {timestamps[seg_idx][1]} speaker_{main_spk_idx}")
        elif sum(spk_for_seg) > 0 and cluster_label > -1:
            main_spk_idx = np.argsort(msdd_preds[seg_idx].cpu().numpy())[::-1][0]
            main_speaker_lines.append(f"{timestamps[seg_idx][0]} {timestamps[seg_idx][1]} speaker_{main_spk_idx}")
            pass
    cont_stamps = get_contiguous_stamps(main_speaker_lines)
    maj_labels = merge_stamps(cont_stamps)
    ovl_labels = get_overlap_stamps(cont_stamps, overlap_speaker_list)
    return maj_labels, ovl_labels

def generate_speaker_timestamps(
    clus_labels: List[Union[float, int]], 
    msdd_preds: List[torch.Tensor], 
    threshold: float,
    max_overlap_count: int = 2,
    **params,
) -> Tuple[List[str], List[str]]:
    """
    Generate speaker timestamps from the segmentation information. 

    Args:
        clus_labels (list):
            List containing integer-valued speaker clustering results.
        msdd_preds (list):
            List containing tensors of the predicted sigmoid values.
            Each tensor has shape of: (Session length, estimated number of speakers).
        threshold (float):
            Sigmoid threshold for MSDD output.
        max_overlap_count (int):
            Maximum number of overlap speakers detected. Default is 2.
        params:
            Parameters for generating RTTM output and evaluation. Parameters include:
                infer_overlap (bool): If False, overlap-speech will not be detected.
                use_clus_as_main (bool): Add overlap-speech detection from MSDD to clustering results. If False, only MSDD output
                                         is used for constructing output RTTM files.
                overlap_infer_spk_limit (int): Above this limit, overlap-speech detection is bypassed.
                use_adaptive_thres (bool): Boolean that determines whehther to use adaptive_threshold depending on the estimated
                                           number of speakers.
                max_overlap_spks (int): Maximum number of overlap speakers detected. Default is 2.
                threshold (float): Sigmoid threshold for MSDD output.

    Returns:
        maj_labels (list):
            List containing string-formated single-speaker speech segment timestamps and corresponding speaker labels.
            Example: [..., '551.685 552.77 speaker_1', '552.99 554.43 speaker_0', '554.97 558.19 speaker_0', ...]
        ovl_labels (list):
            List containing string-formated additional overlapping speech segment timestamps and corresponding speaker labels.
            Note that `ovl_labels` includes only overlapping speech that is not included in `maj_labels`.
            Example: [..., '152.495 152.745 speaker_1', '372.71 373.085 speaker_0', '554.97 555.885 speaker_1', ...]
    """
    if len(msdd_preds.shape) == 3: # Multi-channel late-fusion
        model_spk_num = msdd_preds.shape[1]
        vad_mask = (clus_labels > -1)
    elif len(msdd_preds.shape) == 2:
        msdd_preds.squeeze(0)
        model_spk_num = msdd_preds.shape[-1]
    else:
        raise ValueError(f"msdd_preds shape is not correct: {msdd_preds.shape}")
    clus_labels = clus_labels.cpu().numpy().astype(int)
    vad_mask = (clus_labels > -1)
    speaker_assign_mat = np.zeros_like(msdd_preds)
    clustering_assign_mat = np.zeros_like(msdd_preds)
    # Disable the channels that are not active
    spk_time_each = msdd_preds.sum(dim=0)/msdd_preds.sum()
    if params['mask_spks_with_clus']:
        active_spk_inds = np.unique(clus_labels[clus_labels >= 0])
        mask_ch_inds = torch.ones(msdd_preds.shape[1]).bool()
        mask_ch_inds[active_spk_inds] = False
        msdd_preds[:, mask_ch_inds] = 0.0
    # Assign clustering results only to the active vad frames
    clustering_assign_mat[vad_mask, clus_labels[vad_mask]] = 1
    msdd_preds_masked = np.zeros_like(msdd_preds)
    if not params['infer_overlap']:
        max_overlap_count = 1
    else:
        max_overlap_count = min(active_spk_inds.shape[0], max_overlap_count) # If there is one speaker, then max_overlap_count = 1
    msdd_preds_topk_per_seg, logit_gap = get_top_k_for_each_row(msdd_preds, k_count=max_overlap_count, orig_dim=model_spk_num)
    msdd_preds_top1_per_seg, _ = get_top_k_for_each_row(msdd_preds, k_count=1, orig_dim=model_spk_num)
    if not torch.all((msdd_preds_topk_per_seg > 0.0).sum(axis=1) == max_overlap_count):
        raise ValueError(f"Top-k per seg operation with max_overlap_count: {max_overlap_count} is not correct")
    msdd_preds_topk_per_seg[:, spk_time_each < params['overlap_infer_spk_limit']] = 0.0
    msdd_preds_masked[msdd_preds_topk_per_seg >= threshold] = 1.0
    msdd_preds_masked[vad_mask == False, :] = 0 # Mask out non-vad frames
    speaker_assign_mat = msdd_preds_masked.astype(bool)
    msdd_preds_masked_one = np.zeros_like(msdd_preds)
    msdd_preds_topk_per_seg[logit_gap < threshold] = 0.0
    msdd_preds_masked_ovl = msdd_preds_topk_per_seg.cpu().numpy()
    msdd_preds_masked_one[msdd_preds_top1_per_seg > 0.0] = 1.0
    if not np.all(msdd_preds_masked_one.sum(axis=1) == 1) == True:
        raise ValueError(f"msdd_preds_masked_one is not correct")
    speaker_assign_mat = np.logical_or(msdd_preds_masked_one, msdd_preds_masked_ovl).astype(int)
    if params['ts_vad_threshold'] <= 0:
        speaker_assign_mat[vad_mask==False, :] = 0
    else:
        msdd_step_max = torch.max(msdd_preds, dim=1)[0]
        speaker_assign_mat[(msdd_step_max < params['ts_vad_threshold']), :] = 0
    return speaker_assign_mat

def get_top_k_for_each_row(logit_mat, k_count, orig_dim):
    topk_vals, moc_inds = torch.topk(logit_mat, k=k_count, dim=1)
    top_k_mask = F.one_hot(moc_inds.t().flatten(), num_classes=orig_dim)
    if k_count > 1:
        top_k_mask = top_k_mask.reshape(-1, logit_mat.shape[0], orig_dim).sum(dim=0)
        logit_gap = topk_vals[:, 1]/topk_vals[:, 0]
    else:
        logit_gap = torch.zeros_like(topk_vals[:, 0])
    masked_logit_mat = top_k_mask * logit_mat
    return masked_logit_mat, logit_gap
    
def generate_speaker_assignment_intervals(speaker_assign_mat, timestamps):
    model_spk_num = speaker_assign_mat.shape[-1]
    speaker_assignment = [[] for _ in range(model_spk_num)]
    for seg_idx in range(speaker_assign_mat.shape[0]):
        speaker_vec = speaker_assign_mat[seg_idx]
        for spk_idx in range(model_spk_num):
            if speaker_vec[spk_idx]: 
                speaker_assignment[spk_idx].append(timestamps[seg_idx].tolist())
    return speaker_assignment

def generate_diarization_output_lines(speaker_timestamps, model_spk_num): 
    speaker_lines_total = [] 
    for spk_idx in range(model_spk_num):
        ts_invervals = speaker_timestamps[spk_idx]
        merged_ts_intervals = merge_float_intervals(ts_invervals)
        for ts_interval in merged_ts_intervals:
            speaker_lines_total.extend([f"{ts_interval[0]:.3f} {ts_interval[1]:.3f} speaker_{int(spk_idx)}"])
    return speaker_lines_total
        
def get_uniq_id_list_from_manifest(manifest_file: str):
    """Retrieve `uniq_id` values from the given manifest_file and save the IDs to a list.
    """
    uniq_id_list = []
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            if 'uniq_id' in dic and dic['uniq_id'] is not None:
                uniq_id = dic['uniq_id']
            else:
                uniq_id = get_uniqname_from_filepath(dic['audio_filepath'])
            uniq_id_list.append(uniq_id)
    return uniq_id_list


def get_id_tup_dict(uniq_id_list: List[str], test_data_collection, preds_list: List[torch.Tensor]):
    """
    Create session-level dictionary containing data needed to construct RTTM diarization output.

    Args:
        uniq_id_list (list):
            List containing the `uniq_id` values.
        test_data_collection (collections.DiarizationLabelEntity):
            Class instance that is containing session information such as targeted speaker indices, audio filepath and RTTM filepath.
        preds_list (list):
            List containing tensors of predicted sigmoid values.

    Returns:
        session_dict (dict):
            Dictionary containing session-level target speakers data and predicted simoid values in tensor format.
    """
    session_dict = {x: [] for x in uniq_id_list}
    for idx, line in enumerate(test_data_collection):
        # If the manifest file contains multi-channel files for a session, get `uniq_id` value from the `test_data_collection`.
        if isinstance(line.audio_file, list):
            uniq_id = line.uniq_id
        else:
            uniq_id = get_uniqname_from_filepath(line.audio_file)
        session_dict[uniq_id].append([line.target_spks, preds_list[idx]])
    return session_dict


def prepare_split_data(manifest_filepath, _out_dir, multiscale_args_dict, global_rank):
    """
    This function is needed for preparing diarization training data for multiscale diarization decoder (MSDD).
    Prepare multiscale timestamp data for training. Oracle VAD timestamps from RTTM files are used as VAD timestamps.
    In this function, timestamps for embedding extraction are extracted without extracting the embedding vectors.

    Args:
        manifest_filepath (str):
            Input manifest file for creating audio-to-RTTM mapping.
        _out_dir (str):
            Output directory where timestamp json files are saved.

    Returns:
        multiscale_args_dict (dict):
            - Dictionary containing two types of arguments: multi-scale weights and subsegment timestamps for each data sample.
            - Each data sample has two keys: `multiscale_weights` and `scale_dict`.
                - `multiscale_weights` key contains a list containing multiscale weights.
                - `scale_dict` is indexed by integer keys which are scale index.
            - Each data sample is indexed by using the following naming convention: `<uniq_id>_<start time in ms>_<end time in ms>`
                Example: `fe_03_00106_mixed_626310_642300`
    """
    speaker_dir = os.path.join(_out_dir, 'speaker_outputs')

    # Only if this is for the first run of modelPT instance, remove temp folders.
    if global_rank == 0:
        if os.path.exists(speaker_dir):
            shutil.rmtree(speaker_dir)
        os.makedirs(speaker_dir)
    split_audio_rttm_map = audio_rttm_map(manifest_filepath, attach_dur=True)

    # Speech Activity Detection part
    _speaker_manifest_path = os.path.join(speaker_dir, f'oracle_vad_manifest.json')
    logging.info(f"Extracting oracle VAD timestamps and saving at {speaker_dir}")
    if not os.path.exists(_speaker_manifest_path):
        write_rttm2manifest(split_audio_rttm_map, _speaker_manifest_path, include_uniq_id=True)

    multiscale_timestamps_by_scale = {}

    # Segmentation
    for scale_idx, (window, shift) in multiscale_args_dict['scale_dict'].items():
        subsegments_manifest_path = os.path.join(speaker_dir, f'subsegments_scale{scale_idx}.json')
        if not os.path.exists(subsegments_manifest_path):
            # Sub-segmentation for the current scale (scale_idx)
            segments_manifest_to_subsegments_manifest(
                segments_manifest_file=_speaker_manifest_path,
                subsegments_manifest_file=subsegments_manifest_path,
                window=window,
                shift=shift,
                include_uniq_id=True,
            )
            logging.info(
                f"Subsegmentation for timestamp extracted for: scale-{scale_idx} at {subsegments_manifest_path}"
            )
        multiscale_timestamps = extract_timestamps(subsegments_manifest_path)
        multiscale_timestamps_by_scale[scale_idx] = multiscale_timestamps

    multiscale_timestamps_dict = get_timestamps(multiscale_timestamps_by_scale, multiscale_args_dict)
    return multiscale_timestamps_dict


def extract_timestamps(manifest_file: str):
    """
    This method extracts timestamps from segments passed through manifest_file.

    Args:
        manifest_file (str):
            Manifest file containing segmentation information.
    Returns:
        time_stamps (dict):
            Dictionary containing lists of timestamps.
    """
    logging.info(f"Extracting timestamps from {manifest_file} for multiscale subsegmentation.")
    time_stamps = {}
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_name = dic['uniq_id']
            if uniq_name not in time_stamps:
                time_stamps[uniq_name] = []
            start = dic['offset']
            end = start + dic['duration']
            time_stamps[uniq_name].append([start, end])
    return time_stamps

def change_output_dir_names(params, threshold, verbose=True):
    """
    Create output directories for RTTM and JSON files with the MSDD threshold value.
    """
    head, tail = os.path.split(params['out_rttm_dir']) 
    if not os.path.exists(os.path.join(params['out_rttm_dir'], params['system_name'])):
        os.makedirs(os.path.join(head, params['system_name']), exist_ok=True)
    threshold = "" if not verbose else f"{threshold:.2f}"
    params['out_rttm_dir'] = os.path.join(head, params['system_name'], f"pred_rttms_T{threshold}")
    params['out_json_dir'] = os.path.join(head, params['system_name'], f"pred_jsons_T{threshold}")
    if not os.path.exists(params['out_rttm_dir']):
        os.makedirs(params['out_rttm_dir'], exist_ok=True)
    if not os.path.exists(params['out_json_dir']):
        os.makedirs(params['out_json_dir'], exist_ok=True)
    return params

def mixdown_msdd_preds(
    clus_labels: torch.Tensor,
    msdd_preds: torch.Tensor,
    time_stamps: torch.Tensor,
    offset: float,
    threshold: float, 
    vad_params: dict, 
    params: dict,
    ):
    """
    Generate speaker timestamps from the segmentation information. 
    If `use_clus_as_main=True`, use clustering result for main speaker.

    Args:
        clus_labels (torch.Tensor):
            Tensor containing integer-valued speaker clustering results.
        msdd_preds (torch.Tensor):
            Tensor containing the predicted sigmoid values.
            Shape: (Session length, estimated number of speakers).
        time_stamps (torch.Tensor):
            Tensor containing the time stamps of the audio signal.
        offset (float):
            Offset value for the audio signal.
        threshold (float):
            Sigmoid threshold value for MSDD output.
        vad_params (dict):
            Dictionary containing parameters for VAD.
        params (dict):
            Parameters for generating RTTM output and evaluation. Parameters include:
                infer_overlap (bool): If False, overlap-speech will not be detected.
                use_clus_as_main (bool): Add overlap-speech detection from MSDD to clustering results. If False, only MSDD output
                                         is used for constructing output RTTM files.
                overlap_infer_spk_limit (int): Above this limit, overlap-speech detection is bypassed.
                use_adaptive_thres (bool): Boolean that determines whehther to use adaptive_threshold depending on the estimated
                                           number of speakers.
                max_overlap_spks (int): Maximum number of overlap speakers detected. Default is 2.
                threshold (float): Sigmoid threshold for MSDD output.

    Returns:
        spk_ts (list):
            List containing string-formated single-speaker speech segment timestamps and corresponding speaker labels.
            Example: [..., '551.685 552.77 speaker_1', '552.99 554.43 speaker_0', '554.97 558.19 speaker_0', ...]
    """
    if len(msdd_preds.shape) > 2: # Multichannel case
        mc_speaker_assign_mat = []
        if params['mc_late_fusion_mode'].startswith('post'):
            for ch_idx in range(msdd_preds.shape[2]):
                msdd_preds_ch = msdd_preds[:, :, ch_idx]
                speaker_assign_mat = generate_speaker_timestamps(clus_labels, msdd_preds_ch, threshold, **params)
                mc_speaker_assign_mat.append(speaker_assign_mat)
            mc_speaker_assign_mat = np.stack(mc_speaker_assign_mat, axis=2)
            if params['mc_late_fusion_mode'] == 'post_max':
                speaker_assign_mat = np.max(mc_speaker_assign_mat, axis=2) 
            elif params['mc_late_fusion_mode'] == 'post_mean':
                speaker_assign_mat = np.mean(mc_speaker_assign_mat, axis=2) 
        elif params['mc_late_fusion_mode'].startswith('pre'):
            if params['mc_late_fusion_mode'] == 'pre_mean':
                msdd_preds_mixed = msdd_preds.mean(dim=2)
            speaker_assign_mat = generate_speaker_timestamps(clus_labels, msdd_preds_mixed, threshold, **params)
        else:
            raise NotImplementedError(f"mc_late_fusion_mode: {params['mc_late_fusion_mode']} not implemented.")
    else: # Single channel case
        speaker_assign_mat = generate_speaker_timestamps(clus_labels, offset, threshold, **params)
    
    if params['use_ts_vad']:    
        spk_ts = []
        for spk_id in range(speaker_assign_mat.shape[-1]):
            ts_mat = ts_vad_post_processing(speaker_assign_mat[:, spk_id], vad_params, hop_length=params['hop_len_in_cs'])
            ts_mat = ts_mat + offset
            spk_ts.append(ts_mat.tolist())
    else:
        timestamps = timestamps.cpu().numpy()/100.0
        spk_ts = generate_speaker_assignment_intervals(speaker_assign_mat=speaker_assign_mat, timestamps=timestamps)
    return spk_ts

def make_rttm_with_overlap(
    manifest_file_path: str,
    clus_label_dict: Dict[str, List[Union[float, int]]],
    preds_dict: Dict[str, torch.Tensor],
    ms_ts,
    threshold,
    verbose,
    vad_params,
    **params,
):
    """
    Create RTTM files that include detected overlap speech. Note that the effect of overlap detection is only
    notable when RTTM files are evaluated with `ignore_overlap=False` option.

    Args:
        manifest_file_path (str):
            Path to the input manifest file.
        clus_label_dict (dict):
            Dictionary containing subsegment timestamps in float type and cluster labels in integer type.
            Indexed by `uniq_id` string.
        msdd_preds (list):
            List containing tensors of the predicted sigmoid values.
            Each tensor has shape of: (Session length, estimated number of speakers).
        params:
            Parameters for generating RTTM output and evaluation. Parameters include:
                infer_overlap (bool): If False, overlap-speech will not be detected.
            See docstrings of `generate_speaker_timestamps` function for other variables in `params`.

    Returns:
        all_hypothesis (list):
            List containing Pyannote's `Annotation` objects that are created from hypothesis RTTM outputs.
        all_reference
            List containing Pyannote's `Annotation` objects that are created from ground-truth RTTM outputs
    """
    params = change_output_dir_names(params, threshold, verbose)
    AUDIO_RTTM_MAP = audio_rttm_map(manifest_file_path)
    all_hypothesis, all_reference = [], []
    no_references = False
    logging.info(f"Generating RTTM with infer_mode: {params['infer_mode']}")
    with open(manifest_file_path, 'r', encoding='utf-8') as manifest:
        for i, line in tqdm(enumerate(manifest.readlines()), total=len(manifest.readlines()), desc="Generating RTTM"):
            
            uniq_id = get_uniq_id_from_manifest_line(line)
            
            manifest_dic = AUDIO_RTTM_MAP[uniq_id]
            offset = manifest_dic['offset']
            clus_labels = clus_label_dict[uniq_id]
            
            msdd_preds = preds_dict[uniq_id]
            time_stamps = ms_ts[uniq_id][-1] # Last scale (scale_idx=-1) has the time stamps

            if clus_labels.shape[0] < msdd_preds.shape[0]:
                clus_labels = torch.cat([clus_labels, torch.ones(msdd_preds.shape[0]-clus_labels.shape[0]).long()*-1])

            speaker_timestamps = mixdown_msdd_preds(clus_labels, msdd_preds, time_stamps, offset, threshold, vad_params, params)
            hyp_labels = generate_diarization_output_lines(speaker_timestamps=speaker_timestamps, model_spk_num=msdd_preds.shape[1])
            
            hyp_labels = sorted(hyp_labels, key=lambda x: float(x.split()[0]))
            hypothesis = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
            if params['out_rttm_dir']:
                labels_to_rttmfile(hyp_labels, uniq_id, params['out_rttm_dir'])
            if params['out_json_dir']:
                generate_json_output(hyp_labels, uniq_id, params['out_json_dir'], manifest_dic)
            all_hypothesis.append([uniq_id, hypothesis])
            rttm_file = manifest_dic.get('rttm_filepath', None)
            
            if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
                ref_labels = rttm_to_labels(rttm_file)
                # ref_labels = get_partial_ref_labels(pred_labels=hyp_labels, ref_labels=ref_labels)
                reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
                all_reference.append([uniq_id, reference])
            else:
                no_references = True
                all_reference = []
    return all_reference, all_hypothesis

def generate_json_output(hyp_labels, uniq_id, out_json_dir, manifest_dic, decimals=2):
    json_dict_list = []
    json_dic = {"start_time": None, 
                "end_time": None, 
                "speaker": None, 
                "audio_filepath": manifest_dic['audio_filepath'], 
                "words": None, 
                "offset": None, 
                "duration": None, 
                "text": None}
    
    for line in hyp_labels:
        start, end, label = line.split()
        start, end = float(start), float(end)
        json_dic = deepcopy(json_dic)
        json_dic['start_time'] = start
        json_dic['end_time'] = end 
        json_dic['offset'] = start
        json_dic['duration'] = round(end - start, decimals)
        json_dic['speaker'] = label
        json_dict_list.append(json_dic)
       
    write_diarized_segments(outfile_path=os.path.join(out_json_dir, uniq_id + '.json'), json_dict_list=json_dict_list)
    
def embedding_normalize(embs, use_std=False, eps=1e-10):
    """
    Mean and l2 length normalize the input speaker embeddings

    Args:
        embs: embeddings of shape (Batch,emb_size)
    Returns:
        embs: normalized embeddings of shape (Batch,emb_size)
    """
    embs = embs - embs.mean(axis=0)
    if use_std:
        embs = embs / (embs.std(axis=0) + eps)
    embs_l2_norm = np.expand_dims(np.linalg.norm(embs, ord=2, axis=-1), axis=1)
    embs = embs / embs_l2_norm
    return embs

class OnlineSegmentor:
    """
    Online Segmentor for online (streaming) diarizer.
    - The class instances created by this class takes time-series signal from the audio buffer and
      creates subsegments for embedding extraction.
    - Since online segmentation is based on a short audio buffer, the methods in this class extracts
      a few subsegments from the given intervals for the raw time-series signal.

    Attributes:
        frame_start (float):
            Start of the middle chunk
        buffer_start (float):
            Start of the entire buffer
        buffer_end (float):
            End of the entire buffer
        sample_rate (int):
            Sampling rate of the input time-series signal
        cumulative_speech_labels (Tensor):
            Torch tensor matrix containing culmulative VAD (speech activity) timestamps
    """

    def __init__(self, sample_rate: int):
        self.frame_start: float = 0.0
        self.buffer_start: float = 0.0
        self.buffer_end: float = 0.0
        self.sample_rate: int = sample_rate
        self.cumulative_speech_labels: torch.Tensor = torch.tensor([])

    def run_online_segmentation(
        self,
        audio_buffer: torch.Tensor,
        vad_timestamps: torch.Tensor,
        segment_raw_audio: List[torch.Tensor],
        segment_range_ts: List[List[float]],
        segment_indexes: List[int],
        window: float,
        shift: float,
    ):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is pointing at the onset of the t_range popped most recently.

        Frame is in the middle of the buffer.

        |___Buffer___[___________]____________|
        |____________[   Frame   ]____________|

        | <- buffer start
        |____________| <- frame start


        Args:
            audio_buffer (Tensor):
                Tensor containing raw time-series signal
            vad_timestamps (Tensor):
                Tensor containing VAD intervals (start and end timestamps)
            segment_raw_audio (list):
                List containing the previously added tensors of the raw time-series signal segments
            segment_range_ts (list):
                List containing the previously added intervals (start and end timestamps) of each segment
            segment_indexes (list):
                List containing the previously added global integer indicies of the segments from
                start to current cursor
            window (float):
                Window length in second
            shift (float):
                Shift length in second

        Returns:
            segment_raw_audio (list):
                List containing the newly added tensors of the raw time-series signal
            segment_range_ts (list):
                List containing the newly added interval (start and end timestamps) of each segment
            segment_indexes (list):
                List containing the newly added global integer indicies of the segments from
                start to current cursor
        """
        if self.buffer_start >= 0:
            # Check if this is the very first step
            if len(segment_raw_audio) == 0 and vad_timestamps.shape[0] > 0:
                vad_timestamps[0][0] = max(vad_timestamps[0][0], 0.0)
                speech_labels_for_update = vad_timestamps
                self.cumulative_speech_labels = speech_labels_for_update
            else:
                # Calculate a cursor for the update point
                cursor_for_old_segments, cursor_index = get_new_cursor_for_update(self.frame_start, segment_range_ts)

                segment_range_ts = segment_range_ts[:cursor_index]
                segment_raw_audio = segment_raw_audio[:cursor_index]
                segment_indexes = segment_indexes[:cursor_index]

                if not len(segment_raw_audio) == len(segment_range_ts) == len(segment_indexes):
                    raise ValueError("Scale-wise segment information has a mismatch in length.")

                speech_labels_for_update, self.cumulative_speech_labels = get_speech_labels_for_update(
                    self.frame_start,
                    self.buffer_end,
                    self.cumulative_speech_labels,
                    vad_timestamps,
                    cursor_for_old_segments,
                )

            # Collect the timeseries signal from the buffer
            sigs_list, sig_rangel_list, sig_indexes = get_online_subsegments_from_buffer(
                buffer_start=self.buffer_start,
                buffer_end=self.buffer_end,
                sample_rate=self.sample_rate,
                speech_labels_for_update=speech_labels_for_update,
                audio_buffer=audio_buffer,
                segment_indexes=segment_indexes,
                window=window,
                shift=shift,
            )

            segment_raw_audio.extend(sigs_list)
            segment_range_ts.extend(sig_rangel_list)
            segment_indexes.extend(sig_indexes)

        if not len(segment_raw_audio) == len(segment_range_ts) == len(segment_indexes):
            raise ValueError("Segment information has a mismatch in length.")
        return segment_raw_audio, segment_range_ts, segment_indexes