# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""
python3.9 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools

pip3 install torch==1.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install numpy==1.21.2 filelock==3.0.12 packaging==21.0 huggingface_hub==0.0.17 regex==2021.9.24 sacremoses==0.0.45 tokenizers==0.10.3
pip3 install transformers==4.16.2

wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-mono.tar.gz && tar -xvf checkpoints/codegen-350M-mono.tar.gz -C checkpoints/

python3 mtpb_sample.py
"""

import argparse
import json
import os
from pathlib import Path
import random
from time import time

import torch


########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time()

    def __exit__(self, type, value, traceback):
        print(f"{self.desc} took {time()-self.t:.02f}s")


def set_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = True


def cast(model, fp16=True):
    if fp16:
        model.half()
    return model


def write_jsonl(filename, data):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'wb') as f:
        for x in data:
            f.write((json.dumps(x) + "\n").encode("utf-8"))


########################################################################
# model

from transformers import GPT2TokenizerFast
from jaxformer.hf.codegen.modeling_codegen import CodeGenForCausalLM


def create_model(ckpt, fp16=True):
    if fp16:
        return CodeGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
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
    t.padding_side = "left"
    t.pad_token = 50256
    return t


########################################################################
# sample

def sample(
    device,
    model,
    tokenizer,
    prompt,
    pad_token_id,
    num_return_sequences=1,
    temp=0.2,
    top_p=0.95,
    max_length=2048,
    max_gen_length=128,
):

    input_ids = tokenizer(
        prompt,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).input_ids

    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length

    with torch.no_grad():
        input_ids = input_ids.to(device)
        tokens = model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temp,
            max_length=input_ids_len + max_gen_length,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

    return text


def truncate(completion):
    import re
    
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [re.compile(r, re.MULTILINE) for r in ['^#', re.escape('<|endoftext|>'), "^'''", '^"""', '\n\n\n']]

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


def test_truncate():

    assert truncate('\nif len_a > len_b:\n    result = a\nelse:\n    result = b\n\n\n\n#') == '\nif len_a > len_b:\n    result = a\nelse:\n    result = b'




########################################################################
# benchmark

def create_problem_set(problems_path, problem_ids):

    problems = []
    p = Path(problems_path)
    with p.open("r") as f:
        for line in f:
            try:
                prob = json.loads(line)
            except Exception as e:
                print(p)
                print(line)
                raise e
            if not problem_ids:
                problems.append(prob)
            elif int(prob['id']) in problem_ids:
                problems.append(prob)

    return sorted(problems, key=lambda x: int(x["id"]))


def sample_completions(
    sample,
    device,
    model,
    tokenizer,
    n,
    t,
    p,
    pad_token_id,
    max_length,
    set_rng_seed,
    out_file,
    batch_size,
    problem_set,
    max_gen_length=256,
    prefix = "# Import libraries.\n\nimport numpy as np\n\n"
):

    with print_time("sample completions"):

        wrap = lambda prompt: f'# {prompt}\n'

        for i, problem in enumerate(problem_set):

            if os.path.exists(out_file(problem['id'])):
                print(f'skipping problem {problem["id"]}')
                continue

            samples = []

            print('=' * 10)
            print(f'Problem {problem["id"]}')
            print('=' * 10)

            set_rng_seed()

            num_batches = n // batch_size
            remainder = n % batch_size

            for j in range(num_batches + 1 if remainder > 0 else num_batches):

                filled_prompts = [
                    ([p.format(**input) for p in problem["prompts"]], input, output) for (input, output) in zip(problem["inputs"], problem["outputs"])
                ]

                for k, (prompts, input, output) in enumerate(filled_prompts):

                    histories = [prefix for _ in range(batch_size if j != num_batches else remainder)]
                    histories_full = [[prefix] for _ in range(batch_size if j != num_batches else remainder)]

                    for l, prompt in enumerate(prompts):

                        histories = [h + wrap(prompt) for h in histories]
                        histories_full = [h + [wrap(prompt)] for h in histories_full] 

                        completions = sample(
                            device,
                            model,
                            tokenizer,
                            histories,
                            num_return_sequences=1,
                            top_p=p,
                            temp=t,
                            pad_token_id=pad_token_id,
                            max_length=max_length,
                            max_gen_length=problem.get("max_gen_length", max_gen_length),
                        )

                        histories = [h + f"{truncate(c)}\n\n" for h, c in zip(histories, completions)]
                        histories_full = [h + [f"{truncate(c)}\n\n"] for h, c in zip(histories_full, completions)]

                        print('-' * 10)
                        print(l)
                        print('-' * 10)
                        print(histories[0])
                        print('-' * 10)

                    for history, history_full in zip(histories, histories_full):
                        samples.append(
                            {
                                "id": problem["id"],
                                "input": input,
                                "gold_output": output,
                                "completions": history,
                                "prompts_completions": history_full
                            }
                        )

            write_jsonl(out_file(problem['id']), samples)



########################################################################
# main

def main():

    # (0) params
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/codegen-350M-mono")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--out", type=str, default="./results/{file}/{model}/{file}_{model}_{seed}_{p}_{t}_{n}_{batch_size}_{fp16}_[{problem_ids}].jsonl")
    parser.add_argument("--p", type=float, default=0.95)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--problem-ids", nargs="+", type=int, default=[1])
    parser.add_argument("--problem-path", type=str, default="./mtpb.jsonl")

    args = parser.parse_args()

    out = lambda problem_id: args.out.format(file=os.path.basename(__file__), model=args.model, seed=args.seed, p=args.p, t=args.t, n=args.n, batch_size=args.batch_size, fp16=args.fp16, problem_ids=problem_id)

    device = torch.device(args.device)
    rng_deterministic = True

    # (1) env
    
    set_env()

    def bind_set_seed(seed=args.seed):
        set_seed(seed=seed, deterministic=rng_deterministic)

    # (2) load

    with print_time('loading parameters'):
        model = create_model(ckpt=args.model, fp16=args.fp16).to(device)

    with print_time("load tokenization"):
        tokenizer = create_custom_gpt2_tokenizer()


    # (3) sample

    with print_time("sampling"):
        problem_set = create_problem_set(args.problem_path, args.problem_ids)

        print(f'loaded {len(problem_set)} problems')

        sample_completions(
            sample=sample,
            device=device,
            model=model,
            tokenizer=tokenizer,
            n=args.n,
            t=args.t,
            p=args.p,
            pad_token_id=50256,
            max_length=args.max_length,
            set_rng_seed=bind_set_seed,
            out_file=out,
            batch_size=args.batch_size,
            problem_set=problem_set,
        )


if __name__ == "__main__":
    test_truncate()
    main()
