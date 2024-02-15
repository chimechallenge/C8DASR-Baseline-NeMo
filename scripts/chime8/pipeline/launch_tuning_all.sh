
###########################################################################
### YOUR CUSTOMIZED CONFIGURATIONS HERE ###################################
NEMO_ROOT=${PWD}/../../../ # path to your NeMo root
CHECKPOINTS=${PWD}/nemo_experiments/models_ckpt # pre-trained models checkpoints will be downloaded here
OPTUNA_JOB_NAME=optuna-tune-all
EXP_DIR=${PWD}/nemo_experiments/exp_optuna/${OPTUNA_JOB_NAME}
CHIME_DATA_ROOT=/media/data/datasets/chime8_dasr
DOWNLOAD_ROOT=/media/data/datasets/chime8_datasets # put your download folder here
MIXER6_ROOT=/media/data/datasets/chime8_dasr/mixer6 # you have to put yours
DGEN_SPLITS="train,dev" # which datasets splits you want to generate, note that evaluation will be available later.
STAGE=0 # stage 0 downloads the checkpoints and does data generation. inference starts at stage 1.
STOP_STAGE=0
SCENARIOS="[chime6]"
DIAR_CONFIG="chime8-baseline-chime6"
###########################################################################
cd $NEMO_ROOT

SCRIPT_NAME=${NEMO_ROOT}/scripts/chime8/pipeline/tune_all.py
python -c "import kenlm; print('kenlm imported successfully')" || exit 1

CONFIG_PATH=${NEMO_ROOT}/scripts/chime8/pipeline/confs
YAML_NAME="chime_config"

VAD_MODEL_PATH=${CHECKPOINTS}/MarbleNet_frame_VAD_chime7_Acrobat.nemo
MSDD_MODEL_PATH=${CHECKPOINTS}/MSDD_v2_PALO_100ms_intrpl_3scales.nemo
ASR_MODEL_PATH=${CHECKPOINTS}/FastConformerXL-RNNT-chime7-GSS-finetuned.nemo
LM_MODEL_PATH=${CHECKPOINTS}/ASR_LM_chime7_only.kenlm 

SITE_PACKAGES=`$(which python) -c 'import site; print(site.getsitepackages()[0])'`
DECODERS_DIR=/home/heh/codes/nemo-taejin
export KENLM_ROOT=$DECODERS_DIR/decoders/kenlm
export KENLM_LIB=$DECODERS_DIR/decoders/kenlm/build/bin
export PYTHONPATH=$DECODERS_DIR/decoders:$PYTHONPATH
export PYTHONPATH=$SITE_PACKAGES/kenlm-0.2.0-py3.10-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=$NEMO_ROOT:$PYTHONPATH


####### DATA GENERATION AND DOWNLOAD ################
if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
  if [ ! -d $CHECKPOINTS ]; then
    echo "Downloading checkpoints to $CHECKPOINTS"
    mkdir -p $CHECKPOINTS
    git clone https://huggingface.co/chime-dasr/nemo_baseline_models $CHECKPOINTS
  else
    echo "$CHECKPOINTS folder already exists, skipping downloading models checkpoints."
  fi

  if [ ! -d "$MIXER6_ROOT" ]; then
    echo "$MIXER6_ROOT does not exist. Exiting. Mixer 6 data has to be obtained via the LDC.
    Please take a look at https://www.chimechallenge.org/current/task1/data"
    exit
  fi

  if [ -d "$CHIME_DATA_ROOT" ]; then
    echo "$CHIME_DATA_ROOT exists already, skipping data download and generation."
  else
    chime-utils $DOWNLOAD_ROOT $MIXER6_ROOT $CHIME_DATA_ROOT --part $DGEN_SPLITS --download
  fi
fi


if [ ! -d "$CHIME_DATA_ROOT" ]; then
    echo "$CHIME_DATA_ROOT does not exists, did you have generated the data correctly ? Exiting"
    exit
fi

python ${SCRIPT_NAME} --config-path="${CONFIG_PATH}" --config-name="$YAML_NAME" \
    diar_config=${DIAR_CONFIG} \
    stage=${STAGE} \
    stop_stage=${STOP_STAGE} \
    chime_data_root=${CHIME_DATA_ROOT} \
    output_root=${EXP_DIR} \
    scenarios=${SCENARIOS} \
    subsets="[dev]" \
    asr_model_path=${ASR_MODEL_PATH} \
    lm_model_path=${LM_MODEL_PATH} \
    diarizer.vad.model_path=${VAD_MODEL_PATH} \
    diarizer.msdd_model.model_path=${MSDD_MODEL_PATH} \
    diarizer.use_saved_embeddings=True