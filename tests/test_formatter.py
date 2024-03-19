import math
import json
import numpy.testing as nptest
import textgrid

import formatter

class TestFormatter():
    Format = formatter.Formatter()

    def test_to_TextGrid(self):
        for input_fname, _ in self.provide_to_TextGrid():
            with open(input_fname) as f:
                case = json.load(f)
            observed = self.Format.to_TextGrid(case, by_phrase=False)
            
            assert observed.maxTime is not None
            assert len(observed.tiers) > 0

    def provide_to_TextGrid(self):
        return [
                (
                    'tests/data/TestAudio_SnoopDogg_85SouthMedia_WhisperTimestampSegments.json',
                    ''
                ),
            ]
