import os

import wandb

from src.experiments_utils import retrieve_arguments, set_experiment_mode, preprocess_training, maybe_define_wandb_metrics
from src.experiments_utils import generate_slurm, run_in_cluster_mode, run_in_standard_mode

from .path_manager import PathManager
from .custom_slurm_generator import CustomSlurmGenerator


STEP_METRIC = "train_step"

LOSS_METRICS  = ['loss']
SCORE_METRICS = ['rew']
STAGES = ['train', 'eval']


def sweep_trainer(config_dict=None):
    with wandb.init(config=config_dict) as run:
        train_func(config=wandb.config, use_wandb=True, run=run)


def train_func(config, use_wandb, run=None):
    device = preprocess_training(config=config, seed=config.seed, device=config.device)
    maybe_define_wandb_metrics(loss_metrics=LOSS_METRICS, score_metrics=SCORE_METRICS, stages=STAGES, use_wandb=use_wandb, custom_step_metric=STEP_METRIC)
    
    print('training done')
    

def preprocess_quick_test(config):
    return config



def set_wandb_params(use_wandb:bool):
    if not use_wandb:
        return None, None
    wandb_names = dict(entity="", project="")
    metric_goal = {'name':'eval/rew_mean','goal':'maximize'}
    return wandb_names, metric_goal



if __name__=='__main__':

    verbose  = 4
    save_model = False

    quick_test = True

    use_wandb  = False
    use_sweep  = False
    is_offline = False

    exp_cfg = dict(
        exp_id       = 'basic_training',
        seed         = [0],
        device       = 'default',
        save_model   = save_model,
        verbose      = verbose
    )
    optimizer_cfg = dict(
        lr = [1e-6]
    )
    config = {**exp_cfg, **optimizer_cfg}

    arguments = retrieve_arguments()
    mode, names_dict, cluster_name = set_experiment_mode(arguments=arguments)
    wandb_names, metric_goal = set_wandb_params(use_wandb=use_wandb)
    names_dict = {**names_dict, **wandb_names} if use_wandb else None

    filename = os.path.basename(__file__)
    if mode=="generate_slurm":
        generate_slurm(config=config, filename=filename, cluster_name=cluster_name, SlurmGenerator_cls=CustomSlurmGenerator)
    elif mode=="cluster":
        run_in_cluster_mode(train_func=train_func, filename=filename, CONFIGS_PATH=PathManager.CONFIGS, names_dict=names_dict, SlurmGenerator_cls=CustomSlurmGenerator)
    elif mode=="standard":
        run_in_standard_mode(config=config, train_func=train_func, filename=filename,
                             quick_test=quick_test, use_sweep=use_sweep, use_wandb=use_wandb, is_offline=is_offline,
                             names_dict=names_dict, metric_goal=metric_goal,
                             sweep_trainer=sweep_trainer, preprocess_quick_test_func=preprocess_quick_test,
                             SlurmGenerator_cls=CustomSlurmGenerator, wandb_method="grid")
    else:
        raise ValueError(f'Mode {mode} not supported.')