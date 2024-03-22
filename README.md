# fave-asr: Automated transcription of interview data
[![Maturity badge - level 1](https://img.shields.io/badge/Maturity_Level-In_development-yellowgreen)](http://www.jsoftware.us/vol10/31-E004.pdf)
[![PRs Welcome](https://img.shields.io/badge/Pull_Requests-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![GitHub](https://img.shields.io/github/license/Forced-Alignment-and-Vowel-Extraction/fave-asr?color=blue)

![Build status](https://github.com/Forced-Alignment-and-Vowel-Extraction/fave-asr/actions/workflows/build.yml/badge.svg)
[![codecov](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/fave-asr/graph/badge.svg?token=V54YXTIOPQ)](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/fave-asr)
<!-- For the future: Coveralls for codecoverage -->

## HuggingFace models used
Artifical Intelegence models are powerful and in the wrong hands can be dangerous. 
The models used by fave-asr are cost-free, but you need to accept additional terms of use which confirm you will not misuse these powerful tools.

To use these models:
1. On HuggingFace, [create an account](https://huggingface.co/join) or [log in](https://huggingface.co/login)
2. Accept the terms and conditions for [the segmentation model](https://hf.co/pyannote/segmentation)
3. Accept the terms and conditions for [the diarization model](https://hf.co/pyannote/speaker-diarization-3.1)
4. [Create an access token](https://hf.co/settings/tokens) or copy your existing token

Keep track of your token and keep it safe (e.g. don't accidentally upload it to GitHub). 
We suggest creating an environment variable for your token so that you don't need to paste it into your files.

### Creating an environment variable for your token
#### Linux and Mac
1. Open `~/.bashrc` in a text editor
2. At the end of that file, add the following `HF_TOKEN='<your token>' ; export HF_TOKEN` replacing `<your token>` with [your HuggingFace token](https://hf.co/settings/tokens)

#### Windows
If you run windows and know a solution, edit this file and create a pull request!

## Use
This module is in active development. The use documentation may be out of date. Feel free to edit this file with updated instructions and create a pull request.
1. Follow the [instructions on using HuggingFace models](#HuggingFace models used)
2. Download `pipeline.py`
3. Import that file into your project
4. Set `audio_file = <path to your audio file>`
5. Set `hf_token = <your huggingface token from step 1>`
6. Set `model_name = <whisper model name>`, we recommend `"medium.en"` for English data, otherwise `"large"`
7. Set `device = "cpu"` unless you can run on a GPU, then use `"cuda"`
8. Run `results_segments_w_speakers = pipeline.transcribe_and_diarize(audio_file, hf_token, model_name, device)` 

### Other software required
* `ffmpeg`

## Authors
Luís Roque contributed substantially to the main speaker diarization pipeline. Initial modifications to that code were made by Christian Brickhouse for stability and use as part of the fave-asr library. For licensing of the test audio, see the README in that directory.
