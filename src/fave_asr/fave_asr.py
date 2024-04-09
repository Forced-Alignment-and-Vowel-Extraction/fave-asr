# This program is part of fave-asr
# Copyright (C) 2024 Christian Brickhouse and FAVE Contributors
#
# fave-asr is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation as version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Adapted from software written by Luís Roque and licensed under CC-By-4.0
# See https://github.com/luisroque/large_laguage_models/blob/062d3c1d77da3bafa8f52a951eac099480ce3b15/speech2text_whisperai_pyannotate.py
# for the version adapted.
#
# The modifications are distributed under the terms of the GNU GPL v3
# By complying with the terms of the GNU GPLv3 you comply with the terms of CC-By-4.0
# See https://www.gnu.org/licenses/license-list.en.html#ccby

"""This module automates the transcription and diarization of linguistic data."""

from typing import Optional, List, Dict, Any
import warnings

import textgrid
import whisper_timestamped as whisper
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


def transcribe(
        audio_file: str,
        model_name: str,
        device: str = "cpu",
        detect_disfluencies: bool = True
) -> Dict[str, Any]:
    """
    Transcribe an audio file using a whisper model.

    Args:
        audio_file: Path to the audio file to transcribe.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").
        detect_disfluencies: Flag for whether the transcription should include disfluencies, marked with [*]

    Returns:
        A dictionary representing the transcript segments and language code.
    """
    model = whisper.load_model(model_name, device=device)
    audio = whisper.load_audio(audio_file)
    result = whisper.transcribe(
        model, audio_file, detect_disfluencies=detect_disfluencies)

    language_code = result['language']
    return {
        "segments": result["segments"],
        "language_code": language_code,
    }


def align_segments(
    segments: List[Dict[str, Any]],
    language_code: str,
    audio_file: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Align the transcript segments using a pretrained alignment model (Wav2Vec2 by default).

    Args:
        segments: List of transcript segments to align.
        language_code: Language code of the audio file.
        audio_file: Path to the audio file containing the audio data.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A dictionary representing the aligned transcript segments.
    """
    model_a, metadata = load_align_model(
        language_code=language_code, device=device)
    result_aligned = align(segments, model_a, metadata, audio_file, device)
    return result_aligned


def diarize(audio_file: str, hf_token: str) -> Dict[str, Any]:
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_file: Path to the audio file to diarize.
        hf_token: Authentication token for accessing the Hugging Face API.

    Returns:
        A dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
    """
    diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token)
    diarization_result = diarization_pipeline(audio_file)
    return diarization_result

def assign_speakers(
    diarization_result: Dict[str, Any], aligned_segments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Assign speakers to each transcript segment based on the speaker diarization result.

    Args:
        diarization_result: Dictionary of diarized audio file, including speaker embeddings and number of speakers.
        aligned_segments: Dictionary representing the aligned transcript segments.

    Returns:
        A list of dictionaries representing each segment of the transcript, including 
        the start and end times, the spoken text, and the speaker ID.
    """
    # Considering deprecation
    # warnings.warn("Redundant with assign_word_speakers", DeprecationWarning)
    result_segments = assign_word_speakers(
        diarization_result, aligned_segments
    )
    return result_segments


def transcribe_and_diarize(
    audio_file: str,
    hf_token: str,
    model_name: str,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and perform speaker diarization.

    Args:
        audio_file: Path to the audio file to transcribe and diarize.
        hf_token: Authentication token for accessing the Hugging Face API.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A list of dictionaries representing each segment of the transcript, including 
        the start and end times, the spoken text, and the speaker ID.
    """
    transcript = transcribe(audio_file, model_name, device)
    # aligned_segments = align_segments(
    #    transcript["segments"], transcript["language_code"], audio_file, device
    # )
    diarization_result = diarize(audio_file, hf_token)
    results_segments_w_speakers = assign_speakers(
        diarization_result, transcript)

    return results_segments_w_speakers


def to_TextGrid(diarized_transcription, by_phrase=True):
    """
    Convert a diarized transcription dictionary to a TextGrid

    Args:
        diarized_transcription: Output of pipeline.assign_speakers()
        by_phrase: Flag for whether the intervals should be by phrase (True) or word (False)

    Returns:
        A textgrid.TextGrid object populated with the diarized and
        transcribed data. Tiers are by speaker and contain word-level
        intervals not utterance-level.
    """
    minTime = diarized_transcription['segments'][0]['start']
    maxTime = diarized_transcription['segments'][-1]['end']
    tg = textgrid.TextGrid(minTime=minTime, maxTime=maxTime)

    speakers = [x['speaker']
                for x in diarized_transcription['segments'] if 'speaker' in x]
    for speaker in set(speakers):
        tg.append(textgrid.IntervalTier(
            name=speaker, minTime=minTime, maxTime=maxTime))
    # Create a lookup table of tier indices based on the given speaker name
    tier_key = dict((name, index)
                    for index, name in enumerate([x.name for x in tg.tiers]))

    for i in range(len(diarized_transcription['segments'])):
        segment = diarized_transcription['segments'][i]
        # There's no guarantee, weirdly, that a given word's assigned speaker
        # is the same as the speaker assigned to the whole segment. Since
        # the tiers are based on assigned /segment/ speakers, not assigned
        # word speakers, we need to look up the tier in the segment loop
        # not in the word loop. See Issue #7
        if 'speaker' not in segment:
            warnings.warn('No speaker for segment')
            # print(segment)
            continue
        tier_index = tier_key[segment['speaker']]
        tier = tg.tiers[tier_index]
        minTime = segment['start']
        if i+1 == len(diarized_transcription['segments']):
            maxTime = segment['end']
        else:
            maxTime = diarized_transcription['segments'][i+1]['start']
        mark = segment['text']
        if by_phrase:
            tier.add(minTime, maxTime, mark)
            continue
        for word in segment['words']:
            if 'speaker' not in word:
                warnings.warn(
                    'No speaker assigned to word, using phrase-level speaker')
            elif word['speaker'] != segment['speaker']:
                warnings.warn(
                    'Mismatched speaker for word and phrase, using phrase-level speaker')
                # print(word['speaker'],word)
                # print(segment['speaker'],segment)
                # raise ValueError('Word and segment have different speakers')
            minTime = word['start']
            maxTime = word['end']
            mark = word['text']
            tier.add(minTime, maxTime, mark)
    return tg
