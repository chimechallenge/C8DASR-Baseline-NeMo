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

import logging
import os
import time
from multiprocessing import Process
from pathlib import Path

import optuna
import torch
from asr.run_asr import run_asr
from diar.run_diar import run_diarization
from eval.run_chime_eval import run_chime_evaluation
from gss_process.run_gss_process import run_gss_process
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.core.config import hydra_runner


def sample_params(cfg: DictConfig, trial: optuna.Trial):
    def scale_weights(r, K):
        return [r - kvar * (r - 1) / (K - 1) for kvar in range(K)]

    # Diarization Optimization
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.vad.parameters.frame_vad_threshold = trial.suggest_float("frame_vad_threshold", 0.15, 0.7, step=0.02)
    cfg.diarizer.vad.parameters.pad_onset = round(trial.suggest_float("pad_onset", 0.0, 0.2, step=0.01), 2)
    cfg.diarizer.vad.parameters.pad_offset = round(trial.suggest_float("pad_offset", 0.0, 0.2, step=0.01), 2)
    cfg.diarizer.vad.parameters.min_duration_on = round(trial.suggest_float("min_duration_on", 0.2, 0.4, step=0.05), 2)
    cfg.diarizer.vad.parameters.min_duration_off = round(
        trial.suggest_float("min_duration_off", 0.5, 0.95, step=0.05), 2
    )

    cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [0.55]
    cfg.diarizer.msdd_model.parameters.global_average_mix_ratio = trial.suggest_float(
        "global_average_mix_ratio", low=0.1, high=0.95, step=0.05
    )
    cfg.diarizer.msdd_model.parameters.global_average_window_count = trial.suggest_int(
        "global_average_window_count", low=10, high=500, step=20
    )

    cfg.diarizer.clustering.parameters.oracle_num_speakers = False
    cfg.diarizer.clustering.parameters.max_rp_threshold = round(
        trial.suggest_float("max_rp_threshold", low=0.03, high=0.1, step=0.01), 2
    )
    cfg.diarizer.clustering.parameters.sparse_search_volume = trial.suggest_int(
        "sparse_search_volume", low=25, high=25, step=1
    )
    r_value = round(trial.suggest_float("r_value", 0.5, 2.5, step=0.05), 4)
    scale_n = len(cfg.diarizer.speaker_embeddings.parameters.multiscale_weights)
    cfg.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)

    # GSS Optimization
    cfg.gss.mc_mask_min_db = trial.suggest_int("mc_mask_min_db", -160, -160, 20)
    cfg.gss.mc_postmask_min_db = trial.suggest_int("mc_postmask_min_db", -25, -5, 3)
    cfg.gss.bss_iterations = trial.suggest_int("bss_iterations", 5, 5, 5)
    cfg.gss.dereverb_filter_length = trial.suggest_int("dereverb_filter_length", 5, 5, 5)
    cfg.gss.normalize_db = trial.suggest_int("normalize_db", -20, -20, 5)
    cfg.gss.top_k = trial.suggest_int("top_k", 60, 100, 20)

    cfg.gss.dereverb_prediction_delay = trial.suggest_categorical("dereverb_prediction_delay", choices=[3])
    cfg.gss.dereverb_num_iterations = trial.suggest_categorical("dereverb_num_iterations", choices=[5])
    cfg.gss.mc_filter_type = trial.suggest_categorical("mc_filter_type", choices=['pmwf'])
    cfg.gss.mc_filter_postfilter = trial.suggest_categorical("mc_filter_postfilter", choices=['ban'])

    # ASR Optimization
    cfg.asr.rnnt_decoding.beam.lm_alpha = trial.suggest_float(name='lm_alpha', low=0, high=0.4, step=0.1)
    cfg.asr.rnnt_decoding.beam.lm_beam_size = trial.suggest_int(name='lm_beam_size', low=6, high=8, step=1)
    cfg.asr.rnnt_decoding.beam.maes_num_steps = 5
    cfg.asr.rnnt_decoding.beam.maes_alpha = 3
    cfg.asr.rnnt_decoding.beam.maes_gamma = trial.suggest_float(
        name="maes_expansion_gamma", low=0.3, high=5.3, step=0.5
    )
    cfg.asr.rnnt_decoding.beam.maes_beta = 5
    return cfg


def objective(
    trial: optuna.Trial, gpu_id: int, cfg: DictConfig, optuna_output_dir: str, speaker_output_dir: str,
):
    start_time = time.time()
    with Path(optuna_output_dir, f"trial-{trial.number}") as output_dir:
        # with tempfile.TemporaryDirectory(dir=optuna_output_dir, prefix=f"trial-{trial.number}") as output_dir:
        logging.info(f"Start Trial {trial.number} with output_dir: {output_dir}")

        # Set up some configs based on the current trial
        cfg.gpu_id = gpu_id
        cfg.output_root = output_dir
        if cfg.diarizer.use_saved_embeddings:
            cfg.diarizer.speaker_out_dir = speaker_output_dir
        else:
            cfg.diarizer.speaker_out_dir = output_dir
        cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
        cfg.diarizer.msdd_model.parameters.diar_eval_settings = [
            (cfg.diarizer.collar, cfg.diarizer.ignore_overlap),
        ]
        cfg.prepared_manifest_vad_input = os.path.join(cfg.diarizer.out_dir, 'manifest_vad.json')

        # Sample parameters for this trial
        cfg = sample_params(cfg, trial)

        # Run Diarization
        start_time2 = time.time()
        run_diarization(cfg)
        logging.info(f"Diarization time taken for trial {trial.number}: {(time.time() - start_time2)/60:.2f} mins")

        # Run GSS
        start_time2 = time.time()
        logging.info("Running GSS")
        run_gss_process(cfg)
        logging.info(f"GSS time taken for trial {trial.number}: {(time.time() - start_time2)/60:.2f} mins")

        # Run ASR
        start_time2 = time.time()
        logging.info("Running ASR")
        run_asr(cfg)
        logging.info(f"ASR time taken for trial {trial.number}: {(time.time() - start_time2)/60:.2f} mins")

        # Run evaluation
        logging.info("Running evaluation")
        start_time2 = time.time()
        run_chime_evaluation(cfg)
        logging.info(f"Eval time taken for trial {trial.number}: {(time.time() - start_time2)/60:.2f} mins")

        # Compute objective value
        split = cfg.subsets[0] if isinstance(cfg.subsets, ListConfig) else cfg.subsets
        eval_result_file = os.path.join(cfg.eval_output_dir, split, "macro_wer.txt")
        with open(eval_result_file, "r") as f:
            wer = float(f.read().strip())
        logging.info(f"Time taken for trial {trial.number}: {(time.time() - start_time)/60:.2f} mins")
        logging.info(f"Trial {trial.number} SA-WER: {wer}")

        if not cfg.optuna.get("save_gss_output", False):
            os.system(f"rm -rf {cfg.gss_output_dir}")
        if not cfg.optuna.get("save_diar_output", False):
            os.system(f"rm -rf {cfg.diar_base_dir}")
        if not cfg.optuna.get("save_asr_output", False):
            os.system(f"rm -rf {cfg.asr_output_dir}")
        if not cfg.optuna.get("save_eval_output", False):
            os.system(f"rm -rf {cfg.eval_output_dir}")
        return wer


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {cfg.output_root}")

    def optimize(gpu_id=0):
        worker_func = lambda trial: objective(
            trial, gpu_id, cfg.copy(), cfg.output_root, cfg.optuna.speaker_output_dir
        )

        study = optuna.create_study(
            direction="minimize", study_name=cfg.optuna.study_name, storage=cfg.optuna.storage, load_if_exists=True
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler(cfg.optuna.output_log, mode="a"))
        logger.addHandler(logging.StreamHandler())
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        study.optimize(worker_func, n_trials=cfg.optuna.n_trials, show_progress_bar=True)

    processes = []
    if cfg.optuna.n_jobs == -1:
        cfg.optuna.n_jobs = torch.cuda.device_count()
    n_jobs = min(cfg.optuna.n_jobs, torch.cuda.device_count())
    logging.info(f"Running {cfg.optuna.n_trials} trials on {n_jobs} GPUs")

    for i in range(0, n_jobs):
        p = Process(target=optimize, args=(i,))
        processes.append(p)
        p.start()

    for t in processes:
        t.join()

    study = optuna.load_study(study_name=cfg.optuna.study_name, storage=cfg.optuna.storage,)
    logging.info(f"Best SA-WER {study.best_value}")
    logging.info(f"Best Parameter Set: {study.best_params}")


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
