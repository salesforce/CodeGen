# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# models_nl = ['codegen-350M-nl', 'codegen-2B-nl', 'codegen-6B-nl', 'codegen-16B-nl']
# models_pl = ['codegen-350M-multi', 'codegen-2B-multi', 'codegen-6B-multi', 'codegen-16B-multi',
#              'codegen-350M-mono',
#              'codegen-2B-mono', 'codegen-6B-mono', 'codegen-16B-mono']

import os
import re
import time
import random

import torch

from transformers import GPT2TokenizerFast
from aixcoder.codegen.modeling_codegen import CodeGenForCausalLM


########################################################################
# util
class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time() - self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


def cast(model, fp16=True):
    # if fp16:
    #     model.half()
    return model


########################################################################
# model


def create_model(ckpt, fp16=False):
    # if fp16:
    #     return CodeGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # else:
    return CodeGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


#######################################################################
# sample params
MAX_LENGTH_SAMPLE = 512
TOP_P = 0.95
TEMPERATURE = 0.7
NUM_RETURN_SEQUENCES = 3


def sample(
        model,
        tokenizer,
        context,
        pad_token_id,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_length_sample=MAX_LENGTH_SAMPLE,
        max_length=2048
):
    input_ids = tokenizer(
        context,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).input_ids

    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length

    with torch.no_grad():
        input_ids = input_ids.to()

        tokens = model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            max_length=input_ids_len + max_length_sample,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True,
        )

        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

    return text


def truncate(completion):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


class AIXCode:
    def __init__(self, model_name):
        # preamble
        set_env()
        set_seed(42, deterministic=True)

        ckpt = f'/Users/bytedance/githubcode/CodeGen/checkpoints/{model_name}'

        # load
        with print_time(f'{model_name} loading parameters'):
            model = create_model(ckpt=ckpt, fp16=False).to()

        with print_time(f'{model_name} loading tokenizer'):
            tokenizer = create_custom_gpt2_tokenizer()
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = 50256

        self.model = model
        self.tokenizer = tokenizer

    def aixcode(self, context_string):
        # sample
        with print_time(f'{context_string} ... AIXCoding >>>'):
            result = sample(model=self.model,
                            tokenizer=self.tokenizer,
                            context=context_string,
                            pad_token_id=50256,
                            num_return_sequences=NUM_RETURN_SEQUENCES,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_length_sample=MAX_LENGTH_SAMPLE)

            completion1 = result[0]
            completion2 = result[1]
            completion3 = result[2]

            truncation1 = truncate(completion1)
            truncation2 = truncate(completion2)
            truncation3 = truncate(completion3)

            return f'{context_string} {truncation1} \n\n {context_string} {truncation2} \n\n {context_string} {truncation3} \n\n '
