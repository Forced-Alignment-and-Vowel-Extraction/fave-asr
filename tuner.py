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
# The code in class DataCollatorCTCWithPadding is copyright 
# The HuggingFace Inc. team and copied here under the terms of the
# Apache License, Version 2.0. 
# See transformers/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py
#
# This workflow drew heavily from the tutorial by Patrick von Platen
# https://huggingface.co/blog/fine-tune-wav2vec2-english

import json
import random
import re

from datasets import Audio, load_dataset, load_metric
import pandas as pd # Debugging
import numpy as np
import torchaudio
from transformers import Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor
import textgrid

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        #with self.processor.as_target_processor():
        labels_batch = self.processor.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

strings_to_ignore_regex = r'[\,\?\.\!\-\;\:\*\-\"]'
double_paren_notes_regex = r'\(\(.*?\)\)'
small_pause_regex = r'sp'
non_speech_regex = '\{.*?\}'

def remove_strings(all_text):
    all_text = re.sub(strings_to_ignore_regex, '', all_text)
    all_text = re.sub(double_paren_notes_regex, '', all_text)
    all_text = re.sub(small_pause_regex, '', all_text)
    all_text = re.sub(non_speech_regex, '', all_text)
    all_text = re.sub('\s+', ' ', all_text)
    return all_text

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more examples than there are elements in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0,len(dataset)-1)
        while pick in picks:
            pick = random.randint(0,len(dataset)-1)
        picks.append(pick)
    df = pd.DataFrame(dataset[picks])
    print(df)

def extract_all_chars(batch):
    all_text = " ".join(batch['transcription'])
    all_text = remove_strings(all_text)
    #words,specials = extract_specials(all_text)
    #vocab = list(set(" ".join(words)) | set(specials))
    vocab = list(set(" ".join(all_text)))
    return {"vocab": [vocab], "all_text": [all_text]}

def extract_specials(all_text):
    return special_chars(all_text.split(" "))

def special_chars(words):
    special = []
    not_special = []
    for word in words:
        if word == "sp":
            special.append(word)
            continue
        if "(" in word:
            #special.append(word)
            continue
        if "{" in word:
            special.append(word)
            continue
        if '*' in word:
            #print(word)
            continue
        if '-' in word:
            #print(word)
            continue
        not_special.append(word)
    return (not_special, special)

def make_vocab_dict(corpus):
    vocabs = corpus.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=corpus.column_names["train"])
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    # Add user-defined specials
    #for special_char in specials_set:
    #    vocab_dict[special_char] = len(vocab_dict)
    # Make space more visible
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    # Set blank and unknown tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open('vocab.json','w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate = audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis = -1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references = label_str)
    return {"wer": wer}

def main():
    global processor
    global wer_metric
    test_dir = '/home/cj/Linguistics/california-vowels/fave-asr/VoCal-data/data'
    corpus = load_dataset('audiofolder',data_dir=test_dir)
    corpus.remove_columns(['textgrid'])
    corpus = corpus.cast_column("audio", Audio(sampling_rate=16_000))
    make_vocab_dict(corpus)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,tokenizer=tokenizer)
    corpus = corpus.map(prepare_dataset, remove_columns=corpus.column_names["train"], num_proc=4)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_extractor()
    training_args = TrainingArguments(
        output_dir='./tuner_output',
        group_by_length=True,
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=False,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        push_to_hub=False
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=corpus["train"],
        eval_dataset=corpus["test"],
        tokenizer=processor.feature_extractor,
    )
    return trainer,corpus
