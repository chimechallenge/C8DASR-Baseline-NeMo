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
# script adapted from https://github.com/espnet/espnet/blob/master/egs2/chime7_task1/asr1/local/da_wer_scoring.py

import glob
import json
from pathlib import Path
import os

from omegaconf import DictConfig, OmegaConf
from chime_utils.scoring.meeteval import _wer
from nemo.core.config import hydra_runner
from nemo.utils import logging


def parse_nemo_json(json_file, split_tag=None):
    hyp_segs = []
    with open(json_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "session_id" not in entry:
                audio_file = entry["audio_filepath"]
                if isinstance(audio_file, list):
                    audio_file = audio_file[0]
                session_id = Path(audio_file).stem
                if "_CH" in session_id:
                    session_id = session_id.split("_CH")[0]
                if split_tag:
                    session_id = session_id.split(split_tag)[0]
                entry["session_id"] = session_id
            hyp_segs.append(
                {
                    "speaker": entry["speaker"],
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"],
                    "words": entry["pred_text"],
                    "session_id": entry["session_id"]
                }
            )
    return hyp_segs



def merge_hyp(asr_hyp_folder, scenarios):

    def merge_helper(jsons, scenario):
        out = []
        split_tag = "_" if scenario != "mixer6" else None
        for json_f in jsons:
            utts = parse_nemo_json(json_f, split_tag=split_tag)
            out.extend(utts)

        return out

    for scenario in scenarios:
        c_jsons = glob.glob(os.path.join(asr_hyp_folder, scenario, "*.json"))

        merged_sessions = merge_helper(c_jsons, scenario)
        with open(os.path.join(asr_hyp_folder, f"{scenario}.json"), "w") as f:
            json.dump(merged_sessions, f, indent=4)




def run_evaluation(cfg):
    eval_cfg = OmegaConf.to_container(cfg.eval, resolve=True)
    eval_cfg = DictConfig(eval_cfg)
    for subset in eval_cfg.subsets:

        asr_output_dir = Path(cfg.asr_output_dir)

        merge_hyp(asr_output_dir / subset, eval_cfg.scenarios)

        eval_output_dir = Path(cfg.eval_output_dir)
        eval_output_dir = eval_output_dir / subset
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        _wer(asr_output_dir, Path(eval_cfg.dasr_root), subset, eval_output_dir / "tcpwer", "chime8", eval_cfg.ignore_missing, "tcpWER")
        _wer(asr_output_dir, Path(eval_cfg.dasr_root), subset,
             eval_output_dir / "cpwer", "chime8", eval_cfg.ignore_missing, "cpWER")

@hydra_runner(config_path="../", config_name="chime_config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    run_evaluation(cfg)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
