import json
import numpy.testing as nptest
import os
import pandas

import pipeline

class TestPipeline():
    model_names = ["medium.en"]
    deviced = ["cpu"]
    hf_token = os.environ["HF_TOKEN"]
    language_code = "en"

    def test_transcribe(self):
        for case, ex_fname in self.provide_transcribe():
            observed = pipeline.transcribe(*case)
            with open(ex_fname) as f:
                expected = json.load(f)
            o_transcript = " ".join([x['text'] for x in observed['segments']])
            e_transcript = " ".join([x['text'] for x in expected['segments']])
            assert observed['language_code'] == expected['language_code']
            assert o_transcript == e_transcript

    def test_diarize(self):
        atol = 0.050 # timing can be off by 50ms and still be accepted
        for case, ex_fname in self.provide_diarize():
            observed = pipeline.diarize(*case)
            expected = pandas.read_csv(ex_fname,index_col=0)
            nptest.assert_array_equal(observed.keys(),expected.keys())
            # NB: rtol > 0 means we accept more error at 
            # higher OoM, i.e., as time goes on which we shouldn't do
            nptest.assert_allclose(observed['start'],expected['start'],atol=atol,rtol=0)
            nptest.assert_allclose(observed['end'],expected['end'],atol=atol,rtol=0)
            # Speakers and their order should be stable across runs
            nptest.assert_array_equal(observed['speaker'],expected['speaker'])
            nptest.assert_array_equal(observed['label'],expected['label'])

    def provide_transcribe(self):
        return [
                (
                    [
                        './tests/data/TestAudio_SnoopDogg_85SouthMedia.wav',
                        'medium.en',
                        'cpu'
                    ],
                    './tests/data/TestAudio_SnoopDogg_85SouthMedia.json'
                ),
            ]

    def provide_diarize(self):
        return [
                (
                    [
                        './tests/data/TestAudio_SnoopDogg_85SouthMedia.wav',
                        self.hf_token
                    ],
                    './tests/data/TestAudio_SnoopDogg_85SouthMedia.csv'
                ),
            ]
