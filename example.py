
from .src import retrieve_arguments, set_experiment_mode, generate_slurm, run_in_cluster_mode, remove_lists
from .src import SlurmGenerator


class PathManager:
    def __init__(self):
        pass

class CustomSlurmGenerator(SlurmGenerator):
    def adjust_config_to_constraints(config:dict, slurm_kwargs:dict, cluster_name:str):
        pass

def adjust_for_quick_test(config):
    return config

def train_func():
    pass
    



if __name__=='__main__':

    verbose  = 4
    save_model = False

    quick_test = True
    use_wandb  = False
    run_sweep  = False
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
    mode, names_dict = set_experiment_mode(arguments=arguments)
    

    if mode=="generate_slurm":
        generate_slurm(SLURM_PATH:str, cluster_name:str, SlurmGenerator_cls:Type[SlurmGenerator]):
    
    elif mode=="cluster":
        run_in_cluster_mode(train_func=train_func, CONFIGS_PATH:str, SYNC_WANDB_PATH:str, names_dict:str)
    
    elif mode=="standard":
        if quick_test:
            print('\nQuick test!')
            config = adjust_for_quick_test(config=config)
        if not run_sweep:
            config = remove_lists(config)
        
        if use_wandb and run_sweep:
            import wandb
            with wandb.init(config=config) as run:
                train_func(config=wandb.config, use_wandb=True, run=run)