import math
import json
import numpy.testing as nptest
import pytest
import textgrid
import warnings

import fave_asr as Format

class TestFormatter():

    def test_to_TextGrid(self):
        for input_fname, by_phrase in self.provide_to_TextGrid():
            with open(input_fname) as f:
                case = json.load(f)
            observed = Format.to_TextGrid(case, by_phrase=by_phrase)
            
            assert observed.maxTime is not None
            assert len(observed.tiers) > 0

    def test_no_speaker_warning(self):
        for input_fname in self.provide_no_speaker_warning():
            with open(input_fname) as f:
                case = json.load(f)
            with pytest.warns(UserWarning, match="No speaker for segment") as record:
                _ = Format.to_TextGrid(case, by_phrase=False)

    def provide_to_TextGrid(self):
        return [
                (
                    'tests/data/TestAudio_SnoopDogg_85SouthMedia_WhisperTimestampSegments.json',
                    True
                ),
                (
                    'tests/data/TestAudio_SnoopDogg_85SouthMedia_WhisperTimestampSegments.json',
                    False
                ),
            ]

    def provide_no_speaker_warning(self):
        return [
                'tests/data/TestAudio_SnoopDogg_85SouthMedia.json',
            ]
