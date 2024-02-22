
# CHiME-8 DASR Baseline System

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]
# Environment Setup

--- 

- Challenge official website page: https://www.chimechallenge.org/current/task1/index
- If you want to participate please fill this [Google form](https://forms.gle/9NdhZbDEtbto4Bxn6) (one contact person per-team only).

## Prerequisites

- `git`, `pip`, and `bash` installed on your system.
- CUDA 11.8 compatible hardware and drivers installed for `cupy-cuda11x` to work properly.
- [Miniconda](https://docs.anaconda.com/free/miniconda/) or [Anaconda](https://www.anaconda.com/) installed.
- libsndfile1 ffmpeg packages installed (you can test by calling `ffmpeg` in the shell):
  - via apt: `apt-get update && apt-get install -y libsndfile1 ffmpeg`
  - or you can also use conda: `conda install ffmpeg -c conda-forge -y`


##  Installation

Git clone this repository: 

```bash
git clone https://github.com/chimechallenge/C8DASR-Baseline-NeMo
```

Create a new conda environment and install the latest Pytorch stable version (2.2.0):

```bash
cd C8DASR-Baseline-NeMo
conda create --name c8dasr python==3.10.12
conda activate c8dasr
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install locally NeMo using this repository:
```bash
conda activate c8dasr
cd C8DASR-Baseline-NeMo
pip install Cython
./reinstall.sh
```

Then go to this folder and install the additional dependencies for this recipe: 

```bash
cd scripts/chime8 
./installers/install_c8_dependencies.sh
```

✅ You are ready to go ! <br>

⚠️ If you encountered any problems please see the [trouble shooting page](./docs/trouble_shooting.md), 
feel free to raise an issue or [contact us](#reach_us).


### <a id="reach_us">Any Question/Problem ? Reach us !</a>

If you are considering participating or just want to learn more then please join the <a href="https://groups.google.com/g/chime5/">CHiME Google Group</a>. <br>
We have also a [CHiME Slack Workspace][slack-invite], join the `chime-8-dasr` channel there or contact us directly.<br>


## DASR Data Download and Generation

Data generation is handled here using [chime-utils](https://github.com/chimechallenge/chime-utils). <br>
If you are **only interested in obtaining the data** you should use [chime-utils](https://github.com/chimechallenge/chime-utils) directly. <br>

*Data generation and downloading is also done automatically in this recipe in stage 0**. <br> 
You can skip it if you have already the data. <br>
Note that Mixer 6 Speech has to be obtained via LDC. See [official challenge website](https://www.chimechallenge.org/current/task1/data) for how to obtain Mixer 6. <br>
CHiME-6, DiPCo and NOTSOFAR1 will be downloaded automatically.


## 📊 Results

As explained in [official challenge website](https://www.chimechallenge.org/current/task1/index) this year
systems will be ranked according to macro [tcpWER](https://arxiv.org/pdf/2307.11394.pdf) across the 4 scenarios (5 s collar). <br>

```
###############################################################################                                                                                                                                                         
### tcpWER for all Scenario ###################################################                                                                                                                                                         
###############################################################################                                                                                                                                                         
+-----+--------------+--------------+----------+----------+--------------+-------------+-----------------+------------------+------------------+------------------+        
|     | session_id   |   error_rate |   errors |   length |   insertions |   deletions |   substitutions |   missed_speaker |   falarm_speaker |   scored_speaker |        
|-----+--------------+--------------+----------+----------+--------------+-------------+-----------------+------------------+------------------+------------------|        
| dev | chime6       |     0.581619 |    36692 |    63086 |         5443 |       25901 |            5348 |                1 |                0 |                8 |        
| dev | mixer6       |     0.253877 |    23489 |    92521 |         3891 |       10289 |            9309 |                0 |                0 |               70 |        
| dev | dipco        |     0.667171 |    11454 |    17168 |         2197 |        7036 |            2221 |                1 |                0 |                8 |        
| dev | notsofar1    |     0.735385 |   131042 |   178195 |         9731 |      110923 |           10388 |              298 |                0 |              592 |                                                                     
+-----+--------------+--------------+----------+----------+--------------+-------------+-----------------+------------------+------------------+------------------+                                                                     
###############################################################################                                                                                                                                                         
### Macro-Averaged tcpWER for across all Scenario (Ranking Metric) ############                                                                                                                                                         
###############################################################################                                                                                                                                                         
+-----+--------------+                                                                                                                                                                                                                  
|     |   error_rate |                                                                                                                                                                                                                  
|-----+--------------|                                                                                                                                                                                                                  
| dev |     0.559513 |                                                                                                                                                                                                                  
+-----+--------------+  
```

## Reproducing the Baseline Results

### Inference

If you want to perform inference with the pre-trained models. <br>
We provide an end-to-end script for data generation and inference with the [pre-trained baseline NeMo models](https://huggingface.co/chime-dasr/nemo_baseline_models/tree/main): 
`pipeline/launch_inference.sh`. The models are downloaded automatically from [HuggingFace nemo baseline models repo](https://huggingface.co/chime-dasr/nemo_baseline_models/tree/main) <br>
By default, the scripts hereafter will perform inference on dev set of all 4 scenarios: CHiME-6, DiPCo, Mixer 6 and NOTSOFAR1. <br>
To limit e.g. only to CHiME-6 and DiPCo you can pass this option:

`--SCENARIOS "[chime6,dipco]"`

You can also resume the inference at intermediate stages: 
- `--STAGE 0` data download and generation
- `--STAGE 1` diarization
- `--STAGE 2` guided source separation
- `--STAGE 3` ASR
- `--STAGE 4` scoring

## Launching NeMo CHiME-8 Baseline Inference

If you have already generated the data via [chime-utils](https://github.com/chimechallenge/chime-utils) and the data is in `/path/to/chime8_dasr`:

```bash
./launch_inference.sh --CHIME_DATA_ROOT /path/to/chime8_dasr --STAGE 1
```


If you need to generate the data yet.
CHiME-6, DiPCo and NOTSOFAR1 will be downloaded automatically. Ensure you have ~1TB of space in a path of your choice `/your/path/to/download`. <br>
Mixer 6 Speech has to be obtained via LDC and unpacked in a directory of your choice `/your/path/to/mixer6_root`. <br>
Data will be generated in `/your/path/to/chime8_dasr` again choose the most convenient location for you.

```bash
./launch_inference.sh \ 
--CHIME_DATA_ROOT  /path/to/chime8_dasr \
--DOWNLOAD_ROOT /your/path/to/download \
--MIXER6_ROOT /your/path/to/mixer6_root \
--stage 0 
```

### More details about the inference script 

You can also set up these variables directly in the inference script in `pipeline/launch_inference.sh`. 

```bash
CHECKPOINTS=/path/to/nemo_baseline_models
TEMP_DIR=/temp/path/to/chime8_baseline_tempdir
CHIME_DATA_ROOT=/path/to/chime8_official_cleaned
DOWNLOAD_ROOT=/raid/users/popcornell/chime8datasets # put your download folder here
MIXER6_ROOT=/raid/users/popcornell/mixer6 # you have to put yours
SCENARIOS="[mixer6,chime6,dipco,notsofar1]"
STAGE=0
STOP_STAGE=100
```

- `CHECKPOINTS` Points to the directory `nemo_baseline_models`, where you clone the model repository with `git clone https://huggingface.co/chime-dasr/nemo_baseline_models`.
- `TEMP_DIR` Designated as a space for storing the intermediate processing data (embeddings, manifest files, audio files, etc.).
- `SCENARIOS` Specifies scenario you want to perform inference on in list format.
- `CHIME_DATA_ROOT` Refers to the root directory of the generated CHiME-8 DASR data. If it does not exist and STAGE==0, the directory will be created. 
- `DOWNLOAD_ROOT` The location where downloaded CHiME-8 DASR data will be stored. This applies only if you want to generate data at STAGE 0, if you have it already, you can skip it. 
- `MIXER6_ROOT` Points to the root folder for the extracted Mixer 6 Speech dataset obtained via LDC (see [DASR data page](https://www.chimechallenge.org/current/task1/data)).
- `STAGE` Defines the stage of the pipeline you want to start from:(1: Diarization, 2: GSS, 3: ASR, 4: Evaluation)
- `STOP_STAGE` Defines the stage of the pipeline you want to stop. If `STAGE` == `STOP_STAGE` only one stage will be performed (e.g. for 1 you will only do diarization).


For reference, the generated `CHIME_DATA_ROOT` should look like: 

```
.
├── chime6
│   ├── audio
│   ├── devices
│   ├── transcriptions
│   ├── transcriptions_scoring
│   └── uem
├── dipco
│   ├── audio
│   ├── devices
│   ├── transcriptions
│   ├── transcriptions_scoring
│   └── uem
├── mixer6
│   ├── audio
│   ├── devices
│   ├── transcriptions
│   ├── transcriptions_scoring
│   └── uem
└── notsofar1
    ├── audio
    ├── devices
    ├── transcriptions
    ├── transcriptions_scoring
    └── uem
```

While `MIXER6_ROOT` should look like: 


```
mixer6
├── data 
│   └── pcm_flac 
├── metadata 
│   ├── iv_components_final.csv 
│   ├── mx6_calls.csv 
...
├── splits 
│   ├── dev_a 
│   ├── dev_a.list 
...
└── train_and_dev_files
```



[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>