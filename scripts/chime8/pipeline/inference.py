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

from local.asr.run_asr import run_asr
from local.diar.run_diar import run_diarization
from local.eval.run_eval import run_evaluation
from local.gss.run_gss_process import run_gss_process
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from functools import partial
import numpy as np


def run_stage(stage_num, start_stage=-1, stop_stage=np.inf, skip_stages=None):
    """Simple helper function to avoid boilerplate code for stages"""
    if skip_stages is None:
        skip_stages = []
    if (
        (start_stage <= stage_num)
        and (stop_stage >= stop_stage)
        and (stage_num not in skip_stages)
    ):
        return True
    else:
        return False


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))

    run_stage_flag = partial(
        run_stage,
        start_stage=cfg.stage,
        stop_stage=cfg.stop_stage,
        skip_stages=cfg.skip_stages,
    )

    if run_stage_flag(1):
        logging.info("Running Diarization")
        run_diarization(cfg)

    if run_stage_flag(2):
        logging.info("Running GSS")
        run_gss_process(cfg)

    if run_stage_flag(3):
        logging.info("Running ASR")
        run_asr(cfg)

    if run_stage_flag(4):
        run_evaluation(cfg)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
