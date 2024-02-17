
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
- libsndfile1 ffmpeg packages installed:
  - `apt-get update && apt-get install -y libsndfile1 ffmpeg`


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
systems will be ranked according to macro tcpWER [5] across the 4 scenarios (5 s collar). <br>

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
`pipeline/launch_inference.sh`. <br>
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
./run.sh --CHIME_DATA_ROOT /path/to/chime8_dasr --STAGE 1
```


If you need to generate the data yet.
CHiME-6, DiPCo and NOTSOFAR1 will be downloaded automatically. Ensure you have ~1TB of space in a path of your choice `/your/path/to/download`. <br>
Mixer 6 Speech has to be obtained via LDC and unpacked in a directory of your choice `/your/path/to/mixer6_root`. <br>
Data will be generated in `/your/path/to/chime8_dasr` again choose the most convenient location for you.

```bash
./run.sh --CHIME_DATA_ROOT  /path/to/chime8_dasr \
--DOWNLOAD_ROOT /your/path/to/download \
--MIXER6_ROOT /your/path/to/mixer6_root \
--stage 0 
```






[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>