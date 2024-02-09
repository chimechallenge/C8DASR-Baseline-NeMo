NEMO_ROOT=/home/heh/codes/nemo-taejin
export PYTHONPATH=$NEMO_ROOT:$PYTHONPATH

OPTUNA_JOB_NAME=optuna-e2e-NOd-dev_chime7
DIAR_CONFIG=chime8-baseline  # name for the diarization model

DIAR_MANIFEST_FILEPATH=/media/data/datasets/chime7_official_cleaned_v2_short1mixer6/mixer6-dev-d03-short1.json
CHIME_DATA_ROOT=/media/data/datasets/chime7_official_cleaned_v2_short1mixer6
CHECKPOINTS_DIR=/media/data2/chime7-challenge/checkpoints/
OUTPUT_DIR=./nemo_experiments/${OPTUNA_JOB_NAME}
SPEAKER_OUTPUT_DIR=/home/heh/codes/nemo-taejin/scripts/chime7/pipeline/example_launcher/nemo_experiments/infer-e2e-t385-tango4j-NOd-dev_chime7/diar_results/speaker_outputs

ASR_MODEL_PATH=${CHECKPOINTS_DIR}/rno_chime7_chime6_ft_ptDataSetasrset3_frontend_nemoGSSv1_prec32_layers24_heads8_conv5_d1024_dlayers2_dsize640_bs128_adamw_CosineAnnealing_lr0.0001_wd1e-2_spunigram1024.nemo
LM_MODEL_PATH=${CHECKPOINTS_DIR}/7gram_0001.kenlm
VAD_MODEL_PATH=${CHECKPOINTS_DIR}/frame_vad_chime7_acrobat.nemo
MSDD_MODEL_PATH=${CHECKPOINTS_DIR}/msdd_v2_PALO_bs6_a003_version6_e53.ckpt 


SCENARIOS="[mixer6]"
SUBSETS="[dev]"  # can only optimize one subset

SCRIPT_NAME=${NEMO_ROOT}/scripts/chime7/pipeline/run_full_pipeline_optuna.py
CONFIG_PATH=${NEMO_ROOT}/scripts/chime7/pipeline
YAML_NAME="chime_config.yaml"


SITE_PACKAGES=`$(which python) -c 'import site; print(site.getsitepackages()[0])'`
export KENLM_ROOT=$NEMO_ROOT/decoders/kenlm
export KENLM_LIB=$NEMO_ROOT/decoders/kenlm/build/bin
export PYTHONPATH=$NEMO_ROOT/decoders:$PYTHONPATH
export PYTHONPATH=$SITE_PACKAGES/kenlm-0.2.0-py3.10-linux-x86_64.egg:$PYTHONPATH

CUDA_VISIBLE_DEVICES=1 python ${SCRIPT_NAME} --config-path="${CONFIG_PATH}" --config-name="$YAML_NAME" \
    chime_data_root=${CHIME_DATA_ROOT} \
    output_root=${OUTPUT_DIR} \
    scenarios=$SCENARIOS \
    subsets=$SUBSETS \
    diar_config=$DIAR_CONFIG \
    vad_model_path=${VAD_MODEL_PATH} \
    msdd_model_path=${MSDD_MODEL_PATH} \
    asr_model_path=${ASR_MODEL_PATH} \
    lm_model_path=${LM_MODEL_PATH} \
    diarizer.use_saved_embeddings=true \
    diarizer.manifest_filepath=${DIAR_MANIFEST_FILEPATH} \
    optuna.speaker_output_dir=${SPEAKER_OUTPUT_DIR}
