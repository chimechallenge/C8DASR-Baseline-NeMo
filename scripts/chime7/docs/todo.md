
### TODO: Administrative Section

- [x] Clone internal NVIDIA NeMo Gitlab repo branch to `github.com/tango4j/NeMo/tree/dev/chime7`.  

- [x] Push NeMo manifest creation script to `https://github.com/tango4j/NeMo/tree/dev/chime7/scripts/chime7`.  

- [x] Convert two bash scripts (GSS, RNNT-ASR-BSD) to python based script.  

- [x] Support dataset name "notsofar1" on top of the previous ones: "chime6", "dipco", "mixer6". + Added max. number of speakers

- [ ] Add Whisper text-normalization (as in the original challenge implementation)

- [x] Add Ante's pre-dereverbration to NeMo Multichannel diarization.

- [x] Make inference script in (1) Class based structure (2) And make separate yaml file or dataConfig-class

- [x] Clean and organize environment setting again 

- [ ] Setup and check the training script of NeMo multichannel diarization

- [ ] Setup Optuna script and re-optimize including the new dataset

- [ ] Plug-in 3rd party ASR (Whisper-v3, wavLM) and report the dev/eval-set performance.