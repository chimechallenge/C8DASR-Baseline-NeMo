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

import glob
import logging
import os
import time
from multiprocessing import Process
from pathlib import Path

import optuna
import torch
from local.diar.run_diar import run_diarization
from omegaconf import DictConfig, OmegaConf
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

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
        "sparse_search_volume", low=10, high=25, step=1
    )
    cfg.diarizer.clustering.parameters.sync_score_thres = trial.suggest_float("sync_score_thres", 0.4, 0.95, step=0.02)
    cfg.diarizer.clustering.reclus_aff_thres = trial.suggest_float("reclus_aff_thres", low=0.6, high=0.9, step=0.01)

    r_value = round(trial.suggest_float("r_value", 0.5, 2.5, step=0.05), 4)
    scale_n = len(cfg.diarizer.speaker_embeddings.parameters.multiscale_weights)
    cfg.diarizer.speaker_embeddings.parameters.multiscale_weights = scale_weights(r_value, scale_n)

    return cfg


def objective(
    trial: optuna.Trial, gpu_id: int, cfg: DictConfig, optuna_output_dir: str, speaker_output_dir: str,
):

    with Path(optuna_output_dir, f"trial-{trial.number}") as output_dir:
        logging.info(f"Start Trial {trial.number} with output_dir: {output_dir}, on GPU {gpu_id}")
        with torch.cuda.device(gpu_id):
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
            logging.info(
                f"Diarization time taken for trial {trial.number}: {(time.time() - start_time2) / 60:.2f} mins"
            )

            # Compute objective value
            hyp_rttms = glob.glob(
                os.path.join(cfg.output_root, "diar_outputs", "pred_rttms", "**/*.rttm"), recursive=True
            )
            hyp_rttms = sorted(hyp_rttms, key=lambda x: Path(x).stem)
            gt_rttms = glob.glob(
                os.path.join(cfg.output_root, "diar_outputs", "diar_manifests", "**/*.rttm"), recursive=True
            )
            gt_rttms = sorted(gt_rttms, key=lambda x: Path(x).stem)

            # NOTE: this should be macro !
            der = DiarizationErrorRate(collar=0.5)
            tot_der = []
            for r, h in zip(gt_rttms, hyp_rttms):
                r = load_rttm(r)
                assert len(r.keys()) == 1
                r = r[list(r.keys())[0]]
                h = load_rttm(h)  # pyannote returns a dict, we do not need it
                assert len(h.keys()) == 1
                h = h[list(h.keys())[0]]
                tot_der.append(abs(der(r, h)))

            tot_der = sum(tot_der) / len(tot_der)
            logging.info(f"Finished trial: {trial.number}, macro DER {tot_der}")
            return tot_der


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {cfg.output_root}")

    def optimize(gpu_id):
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

    available_devices = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]

    if cfg.optuna.n_jobs < 0:
        cfg.optuna.n_jobs = len(available_devices)
    n_jobs = min(cfg.optuna.n_jobs, len(available_devices))

    if n_jobs < len(available_devices):
        logging.warning(f"You have {len(available_devices)} but you are only using {n_jobs}.")
    available_devices = available_devices[:n_jobs]

    logging.info(f"Running {cfg.optuna.n_trials} trials on {n_jobs} GPUs, using {available_devices} GPUs")

    for i in range(len(available_devices)):
        p = Process(target=optimize, args=(i,))
        processes.append(p)
        p.start()

    for t in processes:
        t.join()

    study = optuna.load_study(study_name=cfg.optuna.study_name, storage=cfg.optuna.storage)
    logging.info(f"Best DER {study.best_value}")
    logging.info(f"Best Parameter Set: {study.best_params}")


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
