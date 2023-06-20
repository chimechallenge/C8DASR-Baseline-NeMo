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


import torch
import torch.multiprocessing as mp
from megatron.core import parallel_state
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model import MegatronT5AdapterLearningModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState

mp.set_start_method("spawn", force=True)

"""
This is the script to run an Adapter Tuned GPT Model for text generation.

Usage:
    Assume the model has TP=1, PP=1 in the following use cases.
    a. run greedy inference using a base gpt nemo file, and an adapter nemo file:
        python megatron_gpt_ia3_eval.py \
            gpt_model_file=PATH TO GPT MODEL NEMO FILE \
            adapter_model_file=PATH TO ADAPTER MODEL NEMO FILE (generated by training script: ./megatron_gpt_ia3_tuning.py) \
            data_paths=[PATH TO A JSONL FILE CONTAINING PROMPTS], \
            pred_file_path=PATH TO OUTPUT FILE TO DUMP PREDICTIONS
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


@hydra_runner(config_path="conf", config_name="megatron_t5_adapter_inference")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        model_config = MegatronT5AdapterLearningModel.restore_from(
            restore_path=cfg.language_model_path, trainer=trainer, return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    app_state = AppState()
    if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
            app_state.virtual_pipeline_model_parallel_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
        )

    # Load an adapter model,  must be provided in config
    if cfg.get("adapter_model_file", None) is not None and cfg.get("language_model_path", None) is not None:
        # Update frozen GPT model path in case it has changed
        adapter_tuning_cfg = MegatronT5AdapterLearningModel.restore_from(
            cfg.adapter_model_file, trainer=trainer, return_config=True
        )
        with open_dict(adapter_tuning_cfg):
            adapter_tuning_cfg.language_model_path = cfg.language_model_path
            adapter_tuning_cfg.pretrained_language_model_path = cfg.language_model_path
            adapter_tuning_cfg.micro_batch_size = cfg.data.micro_batch_size
            adapter_tuning_cfg.global_batch_size = cfg.data.global_batch_size

        # Now load prompt learning model with frozen gpt model base
        model = MegatronT5AdapterLearningModel.restore_from(
            restore_path=cfg.adapter_model_file, trainer=trainer, override_config_path=adapter_tuning_cfg
        )

    # Or load regular GPT model
    else:
        raise NotImplementedError(
            "This script is meant for inference from an Infused Adapter Tuned T5 Model, config should contain an adapter_model_file and a language_model_path"
        )

    # check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    try:
        model.frozen_model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    test_ds, test_dl = model.build_virtual_prompt_dataset(
        dataset_paths=cfg.data.test_ds,
        batch_size=cfg.data.global_batch_size,
        for_train=False,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)
    response = trainer.predict(model, test_dl)
    print("***************************")
    if cfg.pred_file_path is not None:
        with open(cfg.pred_file_path, "w", encoding="utf-8") as f:
            for batch in response:
                for inp, pred in zip(batch['input_text'], batch['preds_text']):
                    inp = ' '.join(inp.split('\n'))
                    pred = ' '.join(pred.split('\n'))
                    f.write(f'{inp} {pred}\n')
        print("predictions saved to {}".format(cfg.pred_file_path))
    else:
        print(response)
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
