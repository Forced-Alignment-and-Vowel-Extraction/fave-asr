# fave-asr: Automated transcription of interview data
[![Maturity badge - level 1](https://img.shields.io/badge/Maturity_Level-In_development-yellowgreen)](http://www.jsoftware.us/vol10/31-E004.pdf)
[![PRs Welcome](https://img.shields.io/badge/Pull_Requests-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![GitHub](https://img.shields.io/github/license/Forced-Alignment-and-Vowel-Extraction/fave-asr?color=blue)

![PyPI](https://img.shields.io/pypi/v/fave-asr)
![Build status](https://github.com/Forced-Alignment-and-Vowel-Extraction/fave-asr/actions/workflows/build.yml/badge.svg)
[![Build Docs](https://github.com/Forced-Alignment-and-Vowel-Extraction/fave-asr/actions/workflows/quarto_docs.yml/badge.svg)](https://forced-alignment-and-vowel-extraction.github.io/fave-asr/)
[![codecov](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/fave-asr/graph/badge.svg?token=V54YXTIOPQ)](https://codecov.io/gh/Forced-Alignment-and-Vowel-Extraction/fave-asr)
<!-- For the future: Coveralls for codecoverage -->

The FAVE-asr package provides a system for the automated transcription of sociolinguistic interview data on local machines for use by aligners like [FAVE](https://github.com/JoFrhwld/FAVE) or the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/). The package provides functions to label different speakers in the same audio (diarization), transcribe speech, and output TextGrids with phrase- or word-level alignments.

## Example Use Cases

- You want a transcription of an interview for more detailed hand correction.
- You want to transcribe a large corpus and your analysis can tolerate a small error rate.
- You want to make an audio corpus into a text corpus.
- You want to know the number of speakers in an audio file.

For examples on how to use the pacakge, see the [Usage](usage/) pages.

## Installation
To install fave-asr using pip, run the following command in your terminal:

```bash
pip install fave-asr
```

### Other software required
* `ffmpeg` is needed to process the audio. You can [download it from their website](https://ffmpeg.org/download.html)

## Not another transcription service

There are several services which automate the process of transcribing audio, including

- [DARLA CAVE](http://darla.dartmouth.edu/cave)
- [Otter AI](https://otter.ai/)

Unlike other services, `fave-asr` does not require uploading your data to other servers and instead focuses on processing audio on your own computer. Audio data can contain highly confidential information, and uploading this data to other services may not comply with ethical or legal data protection obligations. The goal of `fave-asr` is to serve those use cases where data protection makes local transcription necessary while making the process as seamless as cloud-based transcription services. 

### Example

As an example, we'll transcribe an audio interview of Snoop Dogg by the 85 South Media podcast and output it as a TextGrid.

```{python}
#| output: false
import fave_asr

data = fave_asr.transcribe_and_diarize(
    audio_file = 'usage/resources/SnoopDogg_85SouthMedia.wav',
    hf_token = '',
    model_name = 'small.en',
    device = 'cpu'
    )
tg = fave_asr.to_TextGrid(data)
tg.write('SnoopDogg_85SouthMedia.TextGrid')
```
```{python}
#| echo: false
import io
buffer = io.StringIO()
# textgrid.write() closes the buffer, preventing us from reading it, so trick it
# into calling nothing
close = buffer.close
buffer.close = lambda: None
tg.write(buffer)
print(buffer.getvalue())
```
## Using gated models
Artifical Intelegence models are powerful and in the wrong hands can be dangerous. The models used by fave-asr are cost-free, but you need to accept additional terms of use.

To use these models:
1. On HuggingFace, [create an account](https://huggingface.co/join) or [log in](https://huggingface.co/login)
2. Accept the terms and conditions for [the segmentation model](https://hf.co/pyannote/segmentation)
3. Accept the terms and conditions for [the diarization model](https://hf.co/pyannote/speaker-diarization-3.1)
4. [Create an access token](https://hf.co/settings/tokens) or copy your existing token

Keep track of your token and keep it safe (e.g. don't accidentally upload it to GitHub). 
We suggest creating an environment variable for your token so that you don't need to paste it into your files.

## Creating an environment variable for your token
Storing your tokens as environment variables is a good way to avoid accidentally leaking them. Instead of typing the token into your code and deleting it before you commit, you can use `os.environ["HF_TOKEN"]` to access it from Python instead. This also makes your code more readable since it's obvious what `HF_TOKEN` is while a string of numbers and letters isn't clear.

### Linux and Mac
On Linux and Mac you can store your token in `.bashrc`

1. Open `$HOME/.bashrc` in a text editor
2. At the end of that file, add the following `HF_TOKEN='<your token>' ; export HF_TOKEN` replacing `<your token>` with [your HuggingFace token](https://hf.co/settings/tokens)
3. Add the changes to your current session using `source $HOME/.bashrc`

### Windows
On Windows, use the `setx` command to create an environment variable.
```
setx HF_TOKEN <your token>
```

You need to restart the command line afterwards to make the environment variable available for use. If you try to use the variable in the same window you set the variable, you will run into problems.

### Other software required
* `ffmpeg`

## Authors
Luís Roque contributed substantially to the main speaker diarization pipeline. Initial modifications to that code were made by Christian Brickhouse for stability and use as part of the fave-asr library. For licensing of the test audio, see the README in that directory.
