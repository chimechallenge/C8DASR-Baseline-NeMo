
# CHiME-8 DASR Baseline System

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/chimechallenge.svg?style=social&label=Follow%20%40chimechallenge)](https://twitter.com/chimechallenge)
[![Slack][slack-badge]][slack-invite]

--- 

- Challenge official website page: https://www.chimechallenge.org/current/task1/index
- If you want to participate please fill this [Google form](https://forms.gle/9NdhZbDEtbto4Bxn6) (one contact person per-team only).

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

Data generation and scoring are handled using [CHiME Challenge Utils: github.com/chimechallenge/chime-utils](https://github.com/chimechallenge/chime-utils). <br>

You can also choose to generate 


## Data Generation Only


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



<h3>📩 Contact Us/Stay Tuned</h3>
If you need help, we have also a [CHiME Slack Workspace][slack-invite], you can join the **chime-8-dasr channel** there or contact the organizers directly.
Consider also to join the <a href="https://groups.google.com/g/chime5/">CHiME Google Group</a>. <br>
Please join these two if you plan to participate in this challenge in order to stay updated. 

[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA
[twitter]: https://twitter.com/chimechallenge<h2>References</h2>