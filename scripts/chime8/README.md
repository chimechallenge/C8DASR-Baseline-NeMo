
# CHiME-8 DASR Baseline System

### Prerequisites

- `git`, `pip`, and `bash` installed on your system.
- CUDA 11.8 compatible hardware and drivers installed for `cupy-cuda11x` to work properly.
- [Miniconda](https://docs.anaconda.com/free/miniconda/) or [Anaconda](https://www.anaconda.com/) installed.
- libsndfile1 ffmpeg packages installed:
  - `apt-get update && apt-get install -y libsndfile1 ffmpeg`

## Installation

1. Git clone this repository: 

```bash
git clone https://github.com/chimechallenge/C8DASR_NeMo
```

2. Create a new conda environment and install the latest Pytorch stable version (2.2.0):

```bash
cd C8DASR_NeMo
conda create --name c8dasr python==3.10.12
conda activate c8dasr
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install NeMo: 

```bash
pip install Cython
pip install nemo_toolkit['all']
```

4. Go to this folder and install the additional dependencies: 

```bash
cd scripts/chime8 
./installers/install_c8_dependencies.sh
```

You are ready to go ! <br>

⚠️ If you encountered any problems please see the [trouble shooting page](./docs/trouble_shooting.md), 
feel free to raise an issue or [contact us](#contact). 


# A Step-by-Step Guide for launching NeMo CHiME-8 Baseline

## 0. Data Preparation

Please find the data preparation scripts at chime-utils repository [CHiME Challenge Utils: github.com/chimechallenge/chime-utils](https://github.com/chimechallenge/chime-utils).
You will provide a folder containing datasets and annotations as follows.
```
CHIME_DATA_ROOT=/path/to/chime8_official_cleaned
```

After you finish the dataset preparation, the following needes to be placed in the directory `chime8_official_cleaned`.
```
chime8_official_cleaned/
├── chime6/
├── dipco/
├── mixer6/
└── notsofar1/
```


## 1. Download models for CHiME-8 baseline system from Hugging Face
Visit [Hugging Face CHIME-DASR Repository](https://huggingface.co/chime-dasr/nemo_baseline_models) and download the four model files.   
You need to agree on Hugging Face's terms and conditions to download the files.  

Prepare the four model files as followings in the `CHECKPOINTS` folder:
```
VAD_MODEL_PATH=${CHECKPOINTS}/vad_model.nemo
MSDD_MODEL_PATH=${CHECKPOINTS}/msdd_model.ckpt
ASR_MODEL_PATH=${CHECKPOINTS}/asr_model.nemo
LM_MODEL_PATH=${CHECKPOINTS}/lm_model.kenlm
```


## 2. Setup global varialbes (This is temporary, will be removed in the final format)

Use the main inference script: `<NeMo Root>/scripts/chime8/pipeline/run_full_pipeline.py`

You need to fill the paths to the following variables.
Make sure to setup your CHIME8 Data path, temporary directory with write permissions and NeMo root path where NeMo toolkit is cloned.

```python
NEMO_ROOT="/path/to/NeMo"
CHECKPOINTS="/path/to/checkpoints"
TEMP_DIR="/temp/path/to/chime8_baseline_each1sess"
CHIME_DATA_ROOT="/path/to/chime8_official_cleaned"
SCENARIOS="[mixer6,chime6,dipco]"
DIAR_CONFIG="chime8-baseline-mixer6-short1"
```

## 3. Launch CHiME-8 Baseline 

Before launch the following script, make sure to activate your Conda environment.
```bash
conda activate chime8_baseline
```

Launch the following script after plugging in all the varialbes needed.

```bash
###########################################################################
### YOUR CUSTOMIZED CONFIGURATIONS HERE ###################################
NEMO_ROOT=/path/to/NeMo # Cloned NeMo folder 
CHECKPOINTS=/path/to/checkpoints
TEMP_DIR=/temp/path/to/chime8_baseline_each1sess
CHIME_DATA_ROOT=/path/to/chime8_official_cleaned
SCENARIOS="[mixer6,chime6,dipco,notsofar1]"
DIAR_CONFIG="chime8-baseline-allfour-short1"
MAX_NUM_SPKS=8 # 4 or 8
STAGE=0 # [stage 0] diarization [stage 1] GSS [stage 2] ASR [stage 3] scoring
###########################################################################
cd $NEMO_ROOT
export CUDA_VISIBLE_DEVICES="0"

SCRIPT_NAME=${NEMO_ROOT}/scripts/chime8/pipeline/run_full_pipeline.py
python -c "import kenlm; print('kenlm imported successfully')" || exit 1

CONFIG_PATH=${NEMO_ROOT}/scripts/chime8/pipeline
YAML_NAME="chime_config_t385.yaml"

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
stage=${STAGE} \
diar_config=${DIAR_CONFIG} \
max_num_spks=${MAX_NUM_SPKS} \
chime_data_root=${CHIME_DATA_ROOT} \
output_root=${TEMP_DIR} \
scenarios=${SCENARIOS} \
subsets="[dev]" \
asr_model_path=${ASR_MODEL_PATH} \
lm_model_path=${LM_MODEL_PATH} \
diarizer.vad.model_path=${VAD_MODEL_PATH} \
diarizer.msdd_model.model_path=${MSDD_MODEL_PATH} \
```


## Environment Setup

- This README outlines the steps to set up your environment for the required operations. Please follow these steps in the order presented to ensure a proper setup.
- Environments:
    * [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
    * [CMAKE 3.18](https://cmake.org/)
    * [python 3.10](https://www.python.org/downloads/release/python-3100/)
- **NOTE**: Make sure to install the right version of [PyTorch](https://pytorch.org/) that supports the CUDA version you want.