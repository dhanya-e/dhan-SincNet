<div align="left"><img src="doc/image/espnet_logo1.png" width="550"/></div>

# E2E-SincNet: toward fully end-to-end speech Recognition

This repository is a fork of [ESPnet](https://github.com/espnet/espnet) and contains the code for the paper *E2E-SincNet: toward fully end-to-end speech Recognition*. E2E-SincNet is partially integrated to [ESPnet](https://github.com/espnet/espnet) due to major differences in the input data pipeline. E2E-SincNet will be part of the [SpeechBrain](https://speechbrain.github.io) toolkit. The provided code makes it feasible to reproduce obtained in the paper *E2E-SincNet: toward fully end-to-end speech Recognition*.

# Installation

This repository is an enhanced version of an [ESPnet](https://github.com/espnet/espnet) fork. Therefore, the installation procedure is equivalent to the [ESPnet](https://github.com/espnet/espnet) one, making it easier to deploy SincNet in already existing setups.

# ASR datasets

The current version of E2E-SincNet supports two ASR recipes with the TIMIT and Wall Street Journal datasets. Thus, two recipes named `run_sincnet.sh` are available in `egs/timit/asr1` and `egs/wsj/asr1` to reproduce the results observed on the paper *E2E-SincNet: toward fully end-to-end speech Recognition*.  

# Run the experiments

The proposed integration of SincNet to ESPnet relies on a bridge between the input features preparation of [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi) to the standard ESPnet recipes. Let us consider the TIMIT experiment in this tutorial. Therefore 4 steps are needed:

1. Run the standard Kaldi TIMIT recipe until DNN training (no need to go further). This will create all the files needed by the features pre-processing script.
2. Go to `egs/timit/local` and open `convert_sph_to_wav_kaldiscp_timit.py`. Please modify all the needed path in the latter script with respect to your setup. Then, just call `python convert_sph_to_wav_kaldiscp_timit.py`. This script converts all the TIMIT .WAV files to the correct format for further processing. **PLEASE NOTE THAT THIS SCRIPT DUPLICATES THE WAV FILES SO YOU NEED WRITE ACCESS**
3. Go to `egs/timit/local` and open `save_raw_fea.py`. Please modify all the needed path in the latter script with respect to your setup. More precisely, you will need to modify and call this script 3 times to generate the train/dev and test raw input features (`python convert_sph_to_wav_kaldiscp_timit.py`). This script is from [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi) and will be soon modified so step 1 becomes unnecessary.
4. You're good to finally launch the ESPnet recipe (`run_sincnet.sh`)!

All the configuration files are customizable in the same manner as ESPnet.
