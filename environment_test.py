'''
This script is used to test the environment setup. 
It checks if all the required libraries are installed and if the GPU is available. 
It also checks if the random seeds are fixed properly.
'''

import os
import sys
import time as systime
import random
import argparse
import numpy as np
import pandas as pd
import argparse

import torch
from model import spclt
import model_utils as model_utils
import model_utils.utils_general
import modules as modules
import tasks.macro_modules as macro_modules
import tasks.micro_modules as micro_modules
import tasks.task_utils as task_utils


def main():
    print('--- All the imports are successful ---')
    
    print(f'--- Available cores: {torch.get_num_threads()} available gpus: {torch.cuda.device_count()} ---')
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available(): 
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    print(f'--- Pytorch version: {torch.__version__}, Available threads: {os.cpu_count()} ---')
    
    model_utils.utils_general.fix_seed(131, deterministic=True)  # Below random values in comments are results in the author's machine
    print('Random seed fixed to be 131, testing...')
    print('Python random test:', random.random()) # 0.3154351888034451
    print('Numpy random test:', np.random.rand()) # 0.7809038987924661
    print('Torch random test:', torch.rand(1).item()) # 0.39420515298843384
    if torch.cuda.is_available():
        print('Cudnn random test:', torch.rand(1, device='cuda').item()) # 0.5410190224647522

    print('--- Run again to see if the random values are the same ---')

    sys.exit(0)

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    main()
