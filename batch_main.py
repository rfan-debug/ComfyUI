import itertools
import sys
import copy
import logging
import threading
import heapq
import traceback
import os

import torch

import batch_execution
import folder_paths
import nodes

import comfy.model_management

from comfy.cli_args import args
from utils import cleanup_temp, load_extra_path_config, cuda_malloc_warning

if __name__ == "__main__":
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        print(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()


    q = batch_execution.PromptTaskQueue()

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    nodes.init_custom_nodes()
    cuda_malloc_warning()
