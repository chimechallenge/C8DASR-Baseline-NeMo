
###########################################################################
### YOUR CUSTOMIZED CONFIGURATIONS HERE ###################################
NEMO_ROOT=/path/to/NeMo
CHECKPOINTS=/path/to/checkpoints
TEMP_DIR=/temp/path/to/chime8_baseline_each1sess
CHIME_DATA_ROOT=/path/to/chime7_official_cleaned
SCENARIOS="[mixer6,chime6,dipco]"
DIAR_CONFIG="chime8-baseline-mixer6-short1"
###########################################################################
cd $NEMO_ROOT
export CUDA_VISIBLE_DEVICES="0"

SCRIPT_NAME=${NEMO_ROOT}/scripts/chime7/pipeline/run_full_pipeline.py
python -c "import kenlm; print('kenlm imported successfully')" || exit 1

CONFIG_PATH=${NEMO_ROOT}/scripts/chime7/pipeline
YAML_NAME="chime_config.yaml"

VAD_MODEL_PATH=${CHECKPOINTS}/vad_model.nemo
MSDD_MODEL_PATH=${CHECKPOINTS}/msdd_model.ckpt
ASR_MODEL_PATH=${CHECKPOINTS}/asr_model.nemo
LM_MODEL_PATH=${CHECKPOINTS}/lm_model.kenlm

SITE_PACKAGES=`$(which python) -c 'import site; print(site.getsitepackages()[0])'`
export KENLM_ROOT=$NEMO_ROOT/decoders/kenlm
export KENLM_LIB=$NEMO_ROOT/decoders/kenlm/build/bin
export PYTHONPATH=$NEMO_ROOT/decoders:$PYTHONPATH
export PYTHONPATH=$SITE_PACKAGES/kenlm-0.2.0-py3.10-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=$NEMO_ROOT:$PYTHONPATH

python ${SCRIPT_NAME} --config-path="${CONFIG_PATH}" --config-name="$YAML_NAME" \
    diar_config=${DIAR_CONFIG} \
    chime_data_root=${CHIME_DATA_ROOT} \
    output_root=${TEMP_DIR} \
    scenarios=${SCENARIOS} \
    subsets="[dev]" \
    asr_model_path=${ASR_MODEL_PATH} \
    lm_model_path=${LM_MODEL_PATH} \
    diarizer.vad.model_path=${VAD_MODEL_PATH} \
    diarizer.msdd_model.model_path=${MSDD_MODEL_PATH}