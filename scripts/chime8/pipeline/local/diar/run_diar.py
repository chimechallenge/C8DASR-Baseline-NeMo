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

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models.msdd_v2_models import NeuralDiarizer
from nemo.core.config import hydra_runner
from nemo.utils import logging

from .prepare_nemo_diar_manifest import generate_annotations


def run_diarization(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # Generate Diarization Manifests
    for subset in cfg.subsets:
        generate_annotations(
            data_dir=cfg.chime_data_root,
            subset=subset,
            output_dir=os.path.join(cfg.diar_base_dir, "diar_manifests"),
            scenarios_list=cfg.scenarios,
        )

    # Run Diarization
    for scenario in cfg.scenarios:
        for subset in cfg.subsets:
            cfg.diarizer.manifest_filepath = os.path.join(
                cfg.diar_base_dir, "diar_manifests", scenario, 'mulspk_asr_manifest', f"{scenario}-{subset}.json"
            )
            diarizer_model = NeuralDiarizer(cfg=cfg).to(f"cuda:{cfg.gpu_id}")
            diarizer_model.diarize(verbose=False)

    # Free up memory
    del diarizer_model


@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    logging.info("Running Diarization")
    run_diarization(cfg)

if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
