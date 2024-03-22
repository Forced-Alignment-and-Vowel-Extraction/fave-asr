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
import warnings

import textgrid

class Formatter():
    def __init__(self):
        pass

    def to_TextGrid(self, diarized_transcription, by_phrase=True):
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
        tg = textgrid.TextGrid(minTime=minTime,maxTime=maxTime)

        speakers = [x['speaker'] for x in diarized_transcription['segments'] if 'speaker' in x]
        for speaker in set(speakers):
            tg.append(textgrid.IntervalTier(name=speaker,minTime=minTime,maxTime=maxTime))
        # Create a lookup table of tier indices based on the given speaker name
        tier_key = dict((name,index) for index, name in enumerate([x.name for x in tg.tiers]))

        for i in range(len(diarized_transcription['segments'])):
            segment = diarized_transcription['segments'][i]
            # There's no guarantee, weirdly, that a given word's assigned speaker
            # is the same as the speaker assigned to the whole segment. Since
            # the tiers are based on assigned /segment/ speakers, not assigned 
            # word speakers, we need to look up the tier in the segment loop
            # not in the word loop. See Issue #7
            if 'speaker' not in segment:
                warnings.warn('No speaker for segment')
                #print(segment)
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
                tier.add(minTime,maxTime,mark)
                continue
            for word in segment['words']:
                if 'speaker' not in word:
                    warnings.warn('No speaker assigned to word, using phrase-level speaker')
                elif word['speaker'] != segment['speaker']:
                    warnings.warn('Mismatched speaker for word and phrase, using phrase-level speaker')
                    #print(word['speaker'],word)
                    #print(segment['speaker'],segment)
                    #raise ValueError('Word and segment have different speakers')
                minTime = word['start']
                maxTime = word['end']
                mark = word['text']
                tier.add(minTime,maxTime,mark)
        return tg
