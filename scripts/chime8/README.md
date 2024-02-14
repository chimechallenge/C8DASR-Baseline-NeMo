
# CHiME-8 DASR Baseline System

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]
# Environment Setup

--- 

- Challenge official website page: https://www.chimechallenge.org/current/task1/index
- If you want to participate please fill this [Google form](https://forms.gle/9NdhZbDEtbto4Bxn6) (one contact person per-team only).

### Prerequisites

- `git`, `pip`, and `bash` installed on your system.
- CUDA 11.8 compatible hardware and drivers installed for `cupy-cuda11x` to work properly.
- [Miniconda](https://docs.anaconda.com/free/miniconda/) or [Anaconda](https://www.anaconda.com/) installed.
- libsndfile1 ffmpeg packages installed:
  - `apt-get update && apt-get install -y libsndfile1 ffmpeg`

### Package Installation

1. Git clone this repository: 

```bash
git clone https://github.com/chimechallenge/C8DASR_NeMo
```

2. Create a new conda environment and install the latest Pytorch stable version (2.2.0):
conda activate chime8_baseline
pip uninstall -y 'cupy-cuda118'
pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.1.0"
pip install git+http://github.com/desh2608/gss
pip install optuna
pip install optuna-fast-fanova gunicorn
pip install optuna-dashboard
pip install lhotse==1.14.0
pip install --upgrade jiwer
pip install cmake>=3.18
./run_install_lm.sh "/your/path/to/C8DASR_NeMo"
```

### Detailed Installation Steps for ESPnet and Related Tools

If you have `cupy-cuda118` installed, uninstall it.

```bash
cd C8DASR_NeMo
conda create --name c8dasr python==3.10.12
conda activate c8dasr
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install NeMo: 

```bash
pip install Cython
./reinstall.sh
```

4. Go to this folder and install the additional dependencies: 

```bash
cd scripts/chime8 
./installers/install_c8_dependencies.sh
```

✅ You are ready to go ! <br>

⚠️ If you encountered any problems please see the [trouble shooting page](./docs/trouble_shooting.md), 
feel free to raise an issue or [contact us](#contact).


## Launching NeMo CHiME-8 Baseline Inference

We provide an end-to-end script for data generation and inference with the pre-trained baseline NeMo models: 
`pipeline/launch_inference.sh`.

```bash
./pipeline/launch_inference.sh --DOWNLOAD_ROOT <YOUR_DOWNLOAD_DIR> --MIXER6_ROOT <YOUR_MIXER6_DIR> 

```


## Data Generation

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
DIAR_CONFIG="chime8-baseline-all-4"
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
DIAR_CONFIG="chime8-baseline-all-4"
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


<h3>📩 Contact Us/Stay Tuned</h3>
If you need help, we have also a [CHiME Slack Workspace][slack-invite], you can join the **chime-8-dasr channel** there or contact the organizers directly.
Consider also to join the <a href="https://groups.google.com/g/chime5/">CHiME Google Group</a>. <br>
Please join these two if you plan to participate in this challenge in order to stay updated. 

[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>