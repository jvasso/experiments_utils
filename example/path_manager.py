import os

class PathManager:
    PROJECT_PATH = '.'
    RL_RESULTS   = os.path.join(PROJECT_PATH, 'rl_results')
    SAVED_MODELS = os.path.join(PROJECT_PATH, 'models')
    SLURM        = os.path.join(PROJECT_PATH, 'slurm')
    CONFIGS      = os.path.join(PROJECT_PATH, "configs")
    LOGFILES     = os.path.join(PROJECT_PATH, "logfiles")
    
    SYNC_WANDB = os.path.join(CONFIGS, "sync_wandb.sh")
    
    def __init__(self):
        pass