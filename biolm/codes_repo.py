# coding=utf-8
# Copyright 2020- The Google AI Language Team Authors and The HuggingFace Inc. team and Facebook Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification"""


import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch import nn
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from biolm.utils_classification_long import compute_metrics
from biolm.utils_classification_long import convert_examples_to_features
from biolm.utils_classification_long import output_modes
from biolm.utils_classification_long import processors, stopping_metrics
from transformers.data.processors.utils import InputExample
import dataclasses


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch import nn
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from biolm.utils_classification_long import compute_metrics
from biolm.utils_classification_long import convert_examples_to_features
from biolm.utils_classification_long import output_modes
from biolm.utils_classification_long import processors, stopping_metrics
from transformers.data.processors.utils import InputExample
import dataclasses

from transformers import BertModel


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)

task = 'theraputics'
processor = processors[task]()

output_mode = output_modes[task]
data_dir = 'data/tasks/theraputics/'
model_type="bert"
max_seq_length = 512
model_name_or_path = "/wynton/protected/project/outcome_pred/ucsf_bert_pytorch/512/70k/"


cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )

label_list = processor.get_labels()

examples = (
    processor.get_train_examples(data_dir)
)



tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=False,
            cache_dir=None,
        )

features = convert_examples_to_features(
    examples,
    tokenizer,
    label_list=label_list,
    max_length=max_seq_length,
    output_mode=output_mode,
    pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
    pad_token_segment_id= 0,
)


config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        finetuning_task=task,
        cache_dir= None,
    )

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

batch = dataset[0]
inputs = {"input_ids": batch[0].reshape(1,2560), "attention_mask": batch[1].reshape(1,2560)}
labels = batch[3]
inputs["token_type_ids"] = batch[2].reshape(1,2560)

model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=None,
    )

model = BertModel.from_pretrained(model_name_or_path)
outputs = model(**inputs)

vec_list = []
for i in range(5):
    inputs = {"input_ids": batch[0][512*i:512*(i+1)].reshape(1,512), "attention_mask": batch[1][512*i:512*(i+1)].reshape(1,512)}
    labels = batch[3]
    inputs["token_type_ids"] = batch[2][512*i:512*(i+1)].reshape(1, 512)
    vec_list.append(inputs)

def bert_transform():
    #_model = AutoModelForSequenceClassification.from_pretrained(
    #   model_name_or_path,
    #    from_tf=bool(".ckpt" in model_name_or_path),
    #    config=config,
    #    cache_dir=None,
    #)
    #_model.classifer = nn.Sequential(*list(_model.classifer.children())[:-2])
    #new_classifier = nn.Sequential(*list(_model.children())[:-2])
    model = BertModel.from_pretrained(model_name_or_path)
    return model




class Bert_long(torch.nn.Module):
    def __init__(self, input_size = 5*768, hidden_layer1_size = 2000, hidden_layer2_size = 500, hidden_layer3_size = 100):
        super(Bert_long, self).__init__()
        self.model_1 = bert_transform()
        self.model_2 = bert_transform()
        self.model_3 = bert_transform()
        self.model_4 = bert_transform()
        self.model_5 = bert_transform()
        self.input_size  = input_size
        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.hidden_layer3_size = hidden_layer3_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_layer1_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_layer1_size, self.hidden_layer2_size)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_layer2_size, self.hidden_layer3_size)
        self.relu3 = torch.nn.ReLU()
        self.output= torch.nn.Linear(self.hidden_layer3_size, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        vec_1, vec_2, vec_3, vec_4, vec_5 = x[0], x[1], x[2], x[3], x[4]
        input = torch.cat((self.model_1(**vec_1)[1], self.model_2(**vec_2)[1],self.model_3(**vec_3)[1], self.model_4(**vec_4)[1],
                           self.model_5(**vec_5)[1]),1)
        print(input.shape)
        hidden_1 = self.fc1(input)
        relu_1   = self.relu1(hidden_1)
        hidden_2 = self.fc2(relu_1)
        relu_2 = self.relu2(hidden_2)
        hidden_3 = self.fc3(relu_2)
        relu_3 = self.relu3(hidden_3)
        output= self.output(relu_3)
        output = self.sigmoid(output)
        return output

bertlong = Bert_long()
bertlong(vec_list).shape
# The first step is completed!!!
