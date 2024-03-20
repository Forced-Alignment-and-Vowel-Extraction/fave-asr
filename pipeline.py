# Adapted from software written by Luís Roque and licensed under CC-By-4.0
# See https://github.com/luisroque/large_laguage_models/blob/062d3c1d77da3bafa8f52a951eac099480ce3b15/speech2text_whisperai_pyannotate.py
# for the version copied.
#
# The modifications are distributed under the terms of the GNU GPL v3
# By complying with the terms of the GNU GPLv3 you comply with the terms of CC-By-4.0
# See https://www.gnu.org/licenses/license-list.en.html#ccby
import os
import subprocess
from typing import Optional, List, Dict, Any
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt
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
    result = whisper.transcribe(model, audio_file,detect_disfluencies=detect_disfluencies)

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
    model_a, metadata = load_align_model(language_code=language_code, device=device)
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
        diarization_result: Dictionary representing the diarized audio file, including the speaker embeddings and the number of speakers.
        aligned_segments: Dictionary representing the aligned transcript segments.

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    result_segments = assign_word_speakers(
        diarization_result, aligned_segments
    )
    # Upstream uses this, but it's bugged and I think upstream's upstream has since adopted the
    # output that it tries to create making it redundant
    #
    #results_segments_w_speakers: List[Dict[str, Any]] = []
    #for result_segment in result_segments['segments']:
    #    results_segments_w_speakers.append(
    #        {
    #            "start": result_segment["start"],
    #            "end": result_segment["end"],
    #            "text": result_segment["text"],
    #            "speaker": result_segment["speaker"],
    #            "words": result_segment["words"]
    #        }
    #    )
    return result_segments

def transcribe_and_diarize(
    audio_file: str,
    hf_token: str,
    model_name: str,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Transcribe an audio file and perform speaker diarization to determine which words were spoken by each speaker.

    Args:
        audio_file: Path to the audio file to transcribe and diarize.
        hf_token: Authentication token for accessing the Hugging Face API.
        model_name: Name of the model to use for transcription.
        device: The device to use for inference (e.g., "cpu" or "cuda").

    Returns:
        A list of dictionaries representing each segment of the transcript, including the start and end times, the
        spoken text, and the speaker ID.
    """
    transcript = transcribe(audio_file, model_name, device)
    #aligned_segments = align_segments(
    #    transcript["segments"], transcript["language_code"], audio_file, device
    #)
    diarization_result = diarize(audio_file, hf_token)
    results_segments_w_speakers = assign_speakers(diarization_result, transcript)

    # Print the results in a user-friendly way
    for i, segment in enumerate(results_segments_w_speakers['segments']):
        print(f"Segment {i + 1}:")
        print(f"Start time: {segment['start']:.2f}")
        print(f"End time: {segment['end']:.2f}")
        print(f"Speaker: {segment['speaker']}")
        print(f"Transcript: {segment['text']}")
        print("")

    return results_segments_w_speakers

if __name__ == "__main__":
    model_names = ["medium.en"]
    devices = ["cpu"]
    hf_token = os.environ["HF_TOKEN"]
    language_code = "en"


    audio_file = (
        "/home/cj/Linguistics/california-vowels/data/Test.wav"
    )
    results = {}

    for model_name in model_names:
        results[model_name] = {}
        for device in devices:
            print(f"Testing {model_name} model on {device}")

            start_time = time.time()
            results_segments_w_speakers = transcribe_and_diarize(
                audio_file, hf_token, model_name, device
            )
            end_time = time.time()

            results[model_name][device] = {
                "execution_time": end_time - start_time,
            }

            print(f"Execution time for {model_name} on {device}: {results[model_name][device]['execution_time']:.2f} seconds")
            print("\n")
