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

import contextlib
import glob
import os
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import EncDecCTCModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    setup_model,
    transcribe_partial_audio,
    write_transcription,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class ModelChangeConfig:

    # Sub-config for changes specific to the Conformer Encoder
    conformer: ConformerChangeConfig = ConformerChangeConfig()


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    channel_selector: Optional[
        Union[int, str]
    ] = "average"  # Used to select a single channel from multichannel audio, or use `average` across channels
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    eval_config_yaml: Optional[str] = None  # Path to a yaml file of config of evaluation

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 4
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models)
    compute_timestamps: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)

    # decoder type: ctc or rnnt, can be used to switch between CTC and RNNT decoder for Joint RNNT/CTC models
    decoder_type: Optional[str] = None

    # Use this for model-specific changes before transcription
    model_change: ModelChangeConfig = ModelChangeConfig()

    normalize_db: Optional[float] = None  # Normalize audio to this dB level


def transcribe_speech(cfg):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    results = []
    # Load augmentor from exteranl yaml file which contains eval info, could be extend to other feature such VAD, P&C
    augmentor = None
    if cfg.eval_config_yaml:
        eval_config = OmegaConf.load(cfg.eval_config_yaml)
        augmentor = eval_config.test_ds.get("augmentor")
        logging.info(f"Will apply on-the-fly augmentation on samples during transcription: {augmentor} ")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device: {map_location}")

    asr_model, model_name = setup_model(cfg, map_location)

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # collect additional transcription information
    return_hypotheses = True

    # we will adjust this flag is the model does not support it
    compute_timestamps = cfg.compute_timestamps
    compute_langs = cfg.compute_langs

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if cfg.decoder_type is not None:
            # TODO: Support compute_langs in CTC eventually
            if cfg.compute_langs and cfg.decoder_type == 'ctc':
                raise ValueError("CTC models do not support `compute_langs` at the moment")

            decoding_cfg = cfg.rnnt_decoding if cfg.decoder_type == 'rnnt' else cfg.ctc_decoding
            decoding_cfg.compute_timestamps = cfg.compute_timestamps  # both ctc and rnnt support it
            if 'preserve_alignments' in decoding_cfg:
                decoding_cfg.preserve_alignments = cfg.compute_timestamps
            if 'compute_langs' in decoding_cfg:
                decoding_cfg.compute_langs = cfg.compute_langs

            asr_model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.decoder_type)

        # Check if ctc or rnnt model
        elif hasattr(asr_model, 'joint'):  # RNNT model
            with open_dict(cfg):
                cfg.rnnt_decoding.fused_batch_size = -1
                cfg.rnnt_decoding.compute_timestamps = cfg.compute_timestamps
                cfg.rnnt_decoding.compute_langs = cfg.compute_langs

                if 'preserve_alignments' in cfg.rnnt_decoding:
                    cfg.rnnt_decoding.preserve_alignments = cfg.compute_timestamps

            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            if cfg.compute_langs:
                raise ValueError("CTC models do not support `compute_langs` at the moment.")
            cfg.ctc_decoding.compute_timestamps = cfg.compute_timestamps

            asr_model.change_decoding_strategy(cfg.ctc_decoding)

    if os.path.isdir(cfg.dataset_manifest):
        all_input_manifests = glob.glob(os.path.join(cfg.dataset_manifest, "*.json"))
        logging.info(f"Found {len(all_input_manifests)} manifests in {cfg.dataset_manifest}")
        output_dir = cfg.output_filename if cfg.output_filename else None
    else:
        all_input_manifests = [cfg.dataset_manifest]
        output_dir = os.path.dirname(cfg.output_filename) if cfg.output_filename else None
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Setup decoding config based on model type and decoder_type
    with open_dict(cfg):
        if isinstance(asr_model, EncDecCTCModel) or (
            isinstance(asr_model, EncDecHybridRNNTCTCModel) and cfg.decoder_type == "ctc"
        ):
            cfg.decoding = cfg.ctc_decoding
        else:
            cfg.decoding = cfg.rnnt_decoding

    for input_manifest in all_input_manifests:
        cfg.output_filename = Path(output_dir) / Path(input_manifest).name if output_dir else None
        cfg.dataset_manifest = input_manifest

        logging.info(f"Transcribing audio from {cfg.dataset_manifest}")

        if cfg.dataset_manifest is not None:
            manifest = read_manifest(cfg.dataset_manifest)
            if manifest[0].get("text", None) is None:
                logging.info("No text found in manifest, will set it to 'infer'")
                for idx in range(len(manifest)):
                    manifest[idx]["text"] = "infer"
                write_manifest(cfg.dataset_manifest, manifest)

        # prepare audio filepaths and decide wether it's partical audio
        filepaths, partial_audio = prepare_audio_data(cfg)

        # setup AMP (optional)
        if (
            cfg.amp
            and torch.cuda.is_available()
            and hasattr(torch.cuda, 'amp')
            and hasattr(torch.cuda.amp, 'autocast')
        ):
            logging.info("AMP enabled!\n")
            autocast = torch.cuda.amp.autocast
        else:

            @contextlib.contextmanager
            def autocast():
                yield

        # Compute output filename
        cfg = compute_output_filename(cfg, model_name)

        # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
        if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
            logging.info(
                f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
                f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
            )
            return cfg

        # transcribe audio
        with autocast():
            with torch.no_grad():
                if partial_audio:
                    transcriptions = transcribe_partial_audio(
                        asr_model=asr_model,
                        path2manifest=cfg.dataset_manifest,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        return_hypotheses=return_hypotheses,
                        channel_selector=cfg.channel_selector,
                        augmentor=augmentor,
                    )
                else:
                    transcriptions = asr_model.transcribe(
                        paths2audio_files=filepaths,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        return_hypotheses=return_hypotheses,
                        channel_selector=cfg.channel_selector,
                        augmentor=augmentor,
                        normalize_db=cfg.normalize_db,
                    )

        logging.info(f"Finished transcribing {len(filepaths)} files !")
        logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

        # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
        if type(transcriptions) == tuple and len(transcriptions) == 2:
            transcriptions = transcriptions[0]

        # write audio transcriptions
        output_filename, _ = write_transcription(
            transcriptions,
            cfg,
            model_name,
            filepaths=filepaths,
            compute_langs=compute_langs,
            compute_timestamps=compute_timestamps,
        )
        logging.info(f"Finished writing predictions to {output_filename}!")
        results.append(str(output_filename))
    return results


def run_asr(cfg):
    gss_output_dir = Path(cfg.gss_output_dir, "processed", f"{cfg.diar_config}-{cfg.diar_param}")
    outputs = []
    for scenario in cfg.scenarios:
        for subset in cfg.subsets:
            manifest_dir = gss_output_dir / scenario / subset
            output_dir = Path(cfg.asr_output_dir) / subset / scenario
            output_dir.mkdir(parents=True, exist_ok=True)
            asr_cfg = OmegaConf.to_container(cfg.asr, resolve=True)
            asr_cfg = DictConfig(asr_cfg)
            asr_cfg.dataset_manifest = str(manifest_dir)
            asr_cfg.output_filename = str(output_dir)
            res = transcribe_speech(asr_cfg)
            outputs.append(res)
    return outputs


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    outputs = run_asr(cfg)
    logging.info(f"NeMo manifests saved to: {outputs}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
