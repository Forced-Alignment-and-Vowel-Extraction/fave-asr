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

    def _make_transcript(self, aligned,callback):
        out = []
        for word_dict in aligned['segments']:
            out.append(callback(word_dict))
        return " ".join(out)

    def test_align_segments(self):
        for case, tr_fname, ex_fname in self.provide_align_segments():
            tr = None
            with open(tr_fname,'r') as f:
                tr = json.load(f)
            observed = pipeline.align_segments(tr['segments'],tr['language_code'],*case)
            expected = None
            with open(ex_fname,'r') as f:
                expected = json.load(f)
            o_transcript = self._make_transcript(observed, lambda word_dict: word_dict['text'])
            e_transcript = self._make_transcript(expected, lambda word_dict: word_dict['text'])
            assert o_transcript == e_transcript

    def _assign_funct_case_loader(self, data_provider, callback):
        def mt_callback(word_dict):
            word_speaker = word_dict['speaker']+": "+word_dict['text']
            return word_speaker
        for diarization_fname, aligned_fname, expected_fname in data_provider():
            dr = pandas.read_csv(diarization_fname,index_col=0)
            with open(aligned_fname,'r') as aligned_file:
                al = json.load(aligned_file)
            with open(expected_fname,'r') as expected_file:
                expected = json.load(expected_file)
            observed = callback(dr,al)
            o_transcript = self._make_transcript(observed,mt_callback)
            e_transcript = self._make_transcript(expected,mt_callback)
            assert o_transcript == e_transcript

    def test_assign_word_speakers(self):
        self._assign_funct_case_loader(
                self.provide_assign_word_speakers, pipeline.assign_word_speakers
            )

    def test_assign_speakers(self):
        self._assign_funct_case_loader(
                self.provide_assign_speakers, pipeline.assign_speakers
            )

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

    def provide_align_segments(self):
        return [
                (
                    [
                        './tests/data/TestAudio_SnoopDogg_85SouthMedia.wav',
                        'cpu'
                    ],
                    './tests/data/TestAudio_SnoopDogg_85SouthMedia.json',
                    './tests/data/TestAudio_SnoopDogg_85SouthMedia_aligned.json'
                ),
            ]

    def provide_assign_speakers(self):
        return [
                (
                    './tests/data/TestAudio_SnoopDogg_85SouthMedia.csv',
                    './tests/data/TestAudio_SnoopDogg_85SouthMedia_aligned.json',
                    './tests/data/TestAudio_SnoopDogg_85SouthMedia_segments.json'
                ),
            ]

    def provide_assign_word_speakers(self):
        return self.provide_assign_speakers()
