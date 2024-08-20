import os
import torch
from typing import Type, Callable
from types import SimpleNamespace

import random
import numpy as np

from wandb.sdk.wandb_run import Run

from .slurm_generator import SlurmGenerator
from . import utils_wandb
from . import utils



def set_experiment_mode(arguments):
    mode = ""
    if len(arguments)==0:
        mode = "standard"
    elif len(arguments)==2:
        mode = "generate_slurm"
        if arguments[0] == "generate":
            cluster_name = arguments[1]
        else:
            raise Exception(f'Arguments list {arguments} not supported.')
    elif len(arguments)==3:
        mode = "cluster"
        config_name = arguments[0]
        run_name    = arguments[1]
        group_name  = arguments[2]
    else:
        raise Exception(f'Arguments list {arguments} not supported.')
    names_dict = dict(config_name=config_name, run_name=run_name, group_name=group_name, cluster_name=cluster_name)
    return mode, names_dict



def generate_slurm(SLURM_PATH:str, cluster_name:str, SlurmGenerator_cls:Type[SlurmGenerator]):
    print('\nGenerate slurm!')
    group_name = utils_wandb.generate_group_name(format='format1', cluster_name=cluster_name)
    run_name   = group_name
    slurm_kwargs = utils.load_yaml_file(os.path.join(SLURM_PATH, f'config_{cluster_name}'))
    
    config = SlurmGenerator_cls.adjust_config_to_constraints(config, slurm_kwargs, cluster_name)
    configs_list = utils.dict_of_lists2list_of_dicts(config)
    slurm_generator = SlurmGenerator_cls(configs_list=configs_list,
                                        run_name=run_name,
                                        group_name=group_name,
                                        **slurm_kwargs)
    slurm_generator.generate_config_files(verbose=1)
    slurm_generator.generate_slurm_file(verbose=1)
    slurm_generator.init_log_files(verbose=1)
    slurm_generator.print_instructions()


def single_run(config, train_func:Callable, is_offline, use_wandb:bool, SYNC_WANDB_PATH:str, run_name:str=None, group_name="basic_training") -> Run:
    """
    WARNING: train_func must call preprocess_training at the beginning.
    """
    if use_wandb:
        import wandb
        os.environ["WANDB_MODE"] = "offline" if is_offline else "online"
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        mode = "offline" if is_offline else "online"
        wandb_params = dict(entity="llm4planning2", project="addition", group=group_name)
        run = wandb.init(config=config, sync_tensorboard=False, mode=mode, name=run_name, **wandb_params)
        utils_wandb.update_wandb_sync(run=run, SYNC_WANDB_PATH=SYNC_WANDB_PATH)
    else:
        run = None
    config_object = SimpleNamespace(**config)
    train_func(config=config_object, use_wandb=use_wandb)
    return run


def preprocess_training(config, seed, device):
    utils.set_all_seeds(seed, device=device)
    device = utils.set_reproducible_experiment(seed=config.seed, detect_anomaly=True, device=device)
    return device


def run_in_cluster_mode(train_func:Callable, CONFIGS_PATH:str, SYNC_WANDB_PATH:str, names_dict:str):
    print('\nRun in cluster mode!')
    config_name, run_name, group_name = names_dict["config_name"], names_dict["run_name"], names_dict["group_name"]
    config_file_path = os.path.join(CONFIGS_PATH, config_name)
    single_config = utils.load_yaml_file(file_path=config_file_path)
    run:Run = single_run(config=single_config, train_func=train_func, is_offline=True, use_wandb=True, SYNC_WANDB_PATH=SYNC_WANDB_PATH, run_name=run_name, group_name=group_name)
    run.finish()


def run_sweep(config, sweep_trainer:Callable, method="grid"):
    import wandb
    config   = utils.to_sweep_format(config)
    sweep_id = utils_wandb.create_sweep(parameters=config, method=method)
    wandb.agent(sweep_id, sweep_trainer, count=None)