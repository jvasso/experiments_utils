import random
import string
import time
from datetime import datetime

import pytz

import wandb
from wandb.sdk.wandb_run import Run


# List of adjectives
wandb_adjectives = [
    "frosty", "warm", "gentle", "silent", "falling",
    "ancient", "autumn", "billowing", "broken", "cold",
    "damp", "dark", "dawn", "delicate", "divine",
    "dry", "empty", "floral", "fragrant", "swift",
    "quiet", "white", "roaring", "mystical", "radiant",
    "shimmering", "sleepy", "cozy", "crisp", "sparkling",
    "lonely", "brave", "lively", "proud", "serene",
    "twinkling", "frozen", "peaceful", "bustling"
]

# List of nouns
wandb_nouns = [
    "waterfall", "river", "breeze", "moon", "rain",
    "wind", "sea", "morning", "snow", "lake",
    "sunset", "pine", "shadow", "leaf", "dawn",
    "glitter", "forest", "hill", "cloud", "meadow",
    "stream", "mountain", "field", "star", "flame",
    "night", "galaxy", "ocean", "garden", "path",
    "cave", "valley", "peak", "orchard", "vista",
    "cliff", "lagoon", "palm", "sand"
]


def generate_group_name(format:str=None, cluster_name:str=None):
    """Generates a random name for a run with a three-character suffix."""
    if format is None:
        # Create a local random generator independent of the global random
        local_random = random.Random(time.time())
        adj = local_random.choice(wandb_adjectives)
        noun = local_random.choice(wandb_nouns)
        # Generate a random three-character string using the local random generator
        suffix = ''.join(local_random.choices(string.ascii_lowercase + string.digits, k=3))
        return f"{adj}-{noun}-{suffix}"
    elif format=='format1':
        from .slurm_generator import SlurmGenerator
        assert cluster_name is not None
        cluster_name_short = SlurmGenerator.shorten_cluster_name(cluster_name=cluster_name)
        france_timezone = pytz.timezone('Europe/Paris')
        now = datetime.now(tz=france_timezone)
        formatted_date = now.strftime("%d%B-%Hh%M")
        return cluster_name_short + "_" + formatted_date
    else:
        raise ValueError(f'Format {format} not supported.')



def to_sweep_format(parameters:dict):
    sweep_parameters = {}
    for key,param in parameters.items():
        if isinstance(param, dict):
            raise Exception(f'Dict param not supported')
        elif isinstance(param,list):
            assert len(param) > 0
            if len(param) == 1:
                sweep_parameters[key] = {"value":param[0]}
            else: 
                sweep_parameters[key] = {"values":param}
        else:
            sweep_parameters[key] = {"value":param}
    return sweep_parameters



def initialize_sync_wandb_file(SYNC_WANDB_PATH:str):
    text = "#!/bin/bash\n"
    with open(SYNC_WANDB_PATH, 'w') as script_file:
        script_file.write(text)


def update_wandb_sync(run:Run, SYNC_WANDB_PATH:str):
    dir_name = run.dir.split('/files')[0]
    new_instruction = f"wandb sync {dir_name}"
    with open(SYNC_WANDB_PATH, 'a') as script_file:
        script_file.write("\n" + new_instruction)


def create_sweep(parameters:dict, method:str):
    sweep_config = dict(method=method, metric={'name':'test/rew_mean','goal':'maximize'})
    sweep_config['parameters']=parameters
    sweep_id = wandb.sweep(sweep_config, entity='llm4planning2', project="addition")
    return sweep_id