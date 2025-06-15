import os
import torch
from typing import Type, Callable
from types import SimpleNamespace
import copy

import random
import numpy as np

from wandb.sdk.wandb_run import Run
from collections import defaultdict

from .slurm_generator import SlurmGenerator
from . import utils_wandb
from . import utils



def set_experiment_mode(arguments):
    mode = ""
    names_dict = {}
    cluster_name = None
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
        names_dict = dict(config=config_name, run=run_name, group=group_name)
    else:
        raise Exception(f'Arguments list {arguments} not supported.')
    return mode, names_dict, cluster_name


def single_run(config, train_func:Callable, is_offline:bool, use_wandb:bool, sync_wandb_filepath:str, names_dict:dict) -> Run:
    """
    WARNING: train_func must call preprocess_training at the beginning.
    """
    if use_wandb:
        import wandb
        os.environ["WANDB_MODE"]          = "offline" if is_offline else "online"
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        mode                              = "offline" if is_offline else "online"
        wandb_params = dict(entity=names_dict['entity'], project=names_dict['project'], group=names_dict['group'])
        run_name = names_dict['run'] if 'run' in names_dict.keys() else None
        run = wandb.init(config=config, sync_tensorboard=False, mode=mode, name=run_name, **wandb_params)
        if is_offline:
            utils_wandb.update_wandb_sync(run=run, sync_wandb_filepath=sync_wandb_filepath)
    else:
        run = None
    config_object = SimpleNamespace(**config)
    train_func(config=config_object, use_wandb=use_wandb)
    return run


def preprocess_training(config, seed, device):
    device = utils.set_device(config=config, device=device)
    utils.set_reproducible_experiment(seed=seed, detect_anomaly=True, device=device)
    return device


####################################################################################################################################


def generate_slurm(config, filename:str, cluster_name:str, SlurmGenerator_cls:Type[SlurmGenerator], remove_duplicates=True):
    print('\nGenerate slurm!')
    group_name = utils_wandb.generate_group_name(format='format1', cluster_name=cluster_name)
    run_name   = group_name
    slurm_kwargs = utils.load_yaml_file(os.path.join(SlurmGenerator_cls.SLURM_PATH, f'config_{filename}_{cluster_name}'))
    
    configs_list = utils.dict_of_lists2list_of_dicts(config)
    configs_list_adjusted = [SlurmGenerator_cls.adjust_config_to_constraints(config_element, slurm_kwargs, cluster_name)
                             for config_element in configs_list]
    if remove_duplicates:
        configs_list_adjusted = utils.remove_duplicates(dict_list=configs_list_adjusted)

    slurm_generator = SlurmGenerator_cls(configs_list=configs_list_adjusted,
                                         filename=filename, run_name=run_name, group_name=group_name,
                                         **slurm_kwargs)
    slurm_generator.generate_config_files(verbose=1)
    slurm_generator.generate_slurm_file(verbose=1)
    slurm_generator.init_log_files(verbose=1)
    slurm_generator.print_instructions()


def run_in_cluster_mode(train_func:Callable, filename:str, CONFIGS_PATH:str, names_dict:str, SlurmGenerator_cls:Type[SlurmGenerator]=SlurmGenerator):
    print('\nRun in cluster mode!')
    config_file_path = os.path.join(CONFIGS_PATH, names_dict["config"])
    single_config = utils.load_yaml_file(file_path=config_file_path)
    sync_wandb_filepath = SlurmGenerator_cls.build_sync_wandb_filepath(script_filename=filename, config=single_config)
    run:Run = single_run(config=single_config, train_func=train_func, is_offline=True, use_wandb=True,
                         sync_wandb_filepath=sync_wandb_filepath, names_dict=names_dict)
    run.finish()



def run_in_standard_mode(config, train_func:Callable, filename:str,
                         quick_test:bool, use_sweep:bool, use_wandb:bool, use_special_non_sweep_mode:bool, is_offline:bool,
                         names_dict:dict, metric_goal:dict,
                         sweep_trainer:Callable, preprocess_quick_test_func:Callable, check_config_concistency_func:Callable,
                         preprocess_config_func:Callable,
                         SlurmGenerator_cls:Type[SlurmGenerator]=SlurmGenerator, wandb_method="grid"):
    if use_sweep: assert use_wandb
    assert not (use_sweep and use_special_non_sweep_mode)
    
    config, use_sweep, use_special_non_sweep_mode = check_config_concistency_func(config=config, use_sweep=use_sweep,
                                                                                  use_special_non_sweep_mode=use_special_non_sweep_mode)
    
    if quick_test:
        print('\nQuick test!')
        config = preprocess_quick_test_func(config=config)
    
    if use_wandb and use_sweep:
        import wandb
        config = utils_wandb.to_sweep_format(parameters=config)
        sweep_id = utils_wandb.create_sweep(parameters=config, method=wandb_method, names_dict=names_dict, metric_goal=metric_goal)
        wandb.agent(sweep_id, sweep_trainer, count=None)
    
    elif use_wandb and use_special_non_sweep_mode:
        configs_list = utils.dict_of_lists2list_of_dicts(config)
        configs_list_adjusted = [preprocess_config_func(config_element) for config_element in configs_list]
        configs_list_adjusted = utils.remove_duplicates(dict_list=configs_list_adjusted)
        print(f'\nCombinations to be tested:\n{format_dict_list_string(filter_differing_keys(copy.deepcopy(configs_list_adjusted)))}\n')
        
        names_dict['group'] = utils_wandb.generate_group_name(format='format1', cluster_name='local')
        for config in configs_list_adjusted:
            sync_wandb_filepath = SlurmGenerator_cls.build_sync_wandb_filepath(script_filename=filename, config=config)
            run:Run = single_run(config=config, train_func=train_func, is_offline=is_offline, use_wandb=use_wandb,
                                 sync_wandb_filepath=sync_wandb_filepath, names_dict=names_dict)
            if run is not None:
                run.finish()
        print(f'\n\n### View results at group: {names_dict["group"]}.\n')
    else:
        config = utils.remove_lists(config)
        sync_wandb_filepath = SlurmGenerator_cls.build_sync_wandb_filepath(script_filename=filename, config=config)
        run:Run = single_run(config=config, train_func=train_func, is_offline=is_offline, use_wandb=use_wandb,
                             sync_wandb_filepath=sync_wandb_filepath, names_dict=names_dict)
        if run is not None:
            run.finish()


def filter_differing_keys(dicts):
    if not dicts:
        return []

    # Get all keys from the first dictionary
    keys = dicts[0].keys()
    
    # Identify keys with at least two different values
    differing_keys = {
        key for key in keys 
        if len({d[key] for d in dicts}) > 1
    }

    # Create new list of dicts with only differing keys
    result = [
        {k: d[k] for k in differing_keys} 
        for d in dicts
    ]
    return result


# Function to group formatted dicts by line length and return formatted string
def format_dict_list_string(dict_list):
    grouped = defaultdict(list)

    # Create formatted lines and group them by their string length
    for d in dict_list:
        line = f"    {{{', '.join(f'{repr(k)}: {repr(v)}' for k, v in d.items())}}},"
        grouped[len(line)].append(line)

    # Sort groups by line length and compile output
    lines = ["["]
    for length in sorted(grouped):
        lines.extend(grouped[length])
    lines.append("]")
    
    return "\n".join(lines)