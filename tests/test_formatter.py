import json
import numpy.testing as nptest
import textgrid

import formatter

class TestFormatter():
    Format = formatter.Formatter()

    def test_to_TextGrid(self):
        for input_fname, ex_fname in self.provide_to_TextGrid():
            with open(input_fname) as f:
                case = json.load(f)
            observed = self.Format.to_TextGrid(case)
            
            expected = textgrid.TextGrid()
            expected.read(ex_fname)

            nptest.assert_array_equal(observed,expected)

    def provide_to_TextGrid(self):
        return [
                (
                    'tests/data/TestAudio_SnoopDogg_85SouthMedia_segments.json',
                    'tests/data/TestAudio_SnoopDogg_85SouthMedia.TextGrid'
                ),
            ]
