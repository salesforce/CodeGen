# Minimal example of training the 16B checkpoint on GPU with CPU offloading using deepspeed.

'''
apt install python3.8 python3.8-venv python3.8-dev

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.1 datasets==1.16.1 deepspeed==0.7.0

deepspeed --num_gpus=1 train_deepspeed.py
'''

########################################################################################################
## imports

import os
import argparse
import random
import math

from time import time

import numpy as np

import torch

from transformers import AutoConfig, AutoModelForCausalLM

import deepspeed


########################################################################################################
## args

DEEPSPEED_CONFIG = \
{
    'fp16': {'enabled': True, 'loss_scale': 0, 'loss_scale_window': 1000, 'initial_scale_power': 12, 'hysteresis': 2, 'min_loss_scale': 1},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 1e-05, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}},
    'scheduler': {'type': 'WarmupLR', 'params': {'warmup_min_lr': 0, 'warmup_max_lr': 1e-05, 'warmup_num_steps': 100}},
    'zero_optimization': {
        'stage': 3,
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
        'offload_param': {'device': 'cpu', 'pin_memory': False},
        'overlap_comm': True,
        'contiguous_gradients': True,
        'sub_group_size': 1e9,
        'reduce_bucket_size': 16777216,
        'stage3_prefetch_bucket_size': 15099494.4,
        'stage3_param_persistence_threshold': 40960,
        'stage3_max_live_parameters': 1e9,
        'stage3_max_reuse_distance': 1e9,
        'stage3_gather_fp16_weights_on_model_save': True
    },
    'train_batch_size': 32,
    'train_micro_batch_size_per_gpu': 2,
    'gradient_accumulation_steps': 16,
    'gradient_clipping': 1.0,
    'steps_per_print': 8,
    'wall_clock_breakdown': False,
    'compression_training': {'weight_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {}, 'different_groups': {}}}
}


def create_args(args=argparse.Namespace()):

    args.seed = 42

    args.model = 'Salesforce/codegen-16B-mono'

    args.deepspeed_config = DEEPSPEED_CONFIG

    args.opt_steps_train = 1000

    return args



########################################################################################################
## train

def train(args):

    #######################
    ## preamble

    set_seed(args.seed)


    #######################
    ## model

    print('initializing model')

    config = AutoConfig.from_pretrained(args.model)
    config.gradient_checkpointing = True
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(args.model, config=config)

    model.train()
    # TODO(enijkamp): we need to set this flag twice?
    model.gradient_checkpointing_enable()


    #######################
    ## deepspeed

    print('initializing deepspeed')

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model_engine, optimizer, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model, model_parameters=model_parameters)

    torch.cuda.empty_cache()


    #######################
    ## train

    print('starting training')

    input_ids = torch.randint(low=0, high=10, size=[args.deepspeed_config['train_micro_batch_size_per_gpu'], 1024], dtype=torch.int64).cuda()

    for step in range(args.opt_steps_train+1):

        loss = model_engine(input_ids=input_ids, labels=input_ids).loss

        model_engine.backward(loss)
        model_engine.step()

        print(f'{step} {loss:8.3f}')



########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    return output_dir


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))




########################################################################################################
## main

def main():

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)

    # args
    args = create_args()
    args.output_dir = output_dir
    args.exp_id = exp_id

    # output
    os.makedirs(args.output_dir, exist_ok=True)
    copy_source(__file__, args.output_dir)

    # train
    train(args=args)



if __name__ == '__main__':
    main()