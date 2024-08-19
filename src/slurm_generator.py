from abc import abstractmethod
import os
from typing import List

import yaml

from . import utils
from . import utils_wandb

from wandb.sdk.wandb_run import Run


class SlurmGenerator:

    SEP = "\n"
    
    QOS_DEV = "qos_gpu-dev"
    QOS_T3  = "qos_gpu-t3"
    QOS_T4  = "qos_gpu-t4"

    CLUSTER_JEAN_ZAY = "jean_zay"
    CLUSTER_RUCHE    = "ruche"
    
    def __init__(self, cluster_name:str, configs_list:List[dict], run_name:str, group_name:str,
                 SLURM_PATH:str, CONFIGS_PATH:str, LOGFILES_PATH:str, SYNC_WANDB_PATH:str,
                 time:str, qos:str=None, constraint:str=None, partition:str=None,
                 slurm_filename:str="rl_train"):
        self.cluster      = cluster_name
        self.configs_list = configs_list
        self.run_name     = run_name
        self.group_name   = group_name
        self.time         = time
        self.qos          = qos
        self.constraint   = constraint
        self.partition    = partition

        self.SLURM_PATH      = SLURM_PATH
        self.CONFIGS_PATH    = CONFIGS_PATH
        self.LOGFILES_PATH   = LOGFILES_PATH
        self.SYNC_WANDB_PATH = SYNC_WANDB_PATH

        self.slurm_filename = f'{slurm_filename}_{self.cluster}.sh'
        
        self.num_configs = len(self.configs_list)

        if self.cluster==self.CLUSTER_JEAN_ZAY:
            self.preprocess_jean_zay()
        elif self.cluster==self.CLUSTER_RUCHE:
            self.preprocess_ruche()
        else:
            raise ValueError(f'Cluster name {self.cluster} not supported.')


    def preprocess_jean_zay(self):
        if self.num_configs > 10:
            print(f"\nWARNING: num of configs {self.num_configs} greater than 10.\n")
        else:
            print(f"{self.num_configs} configs to run.")
    

    def preprocess_ruche(self):
        if self.partition in {'gpu','gpu_test'}:
            assert self.num_configs <= 8
        elif self.partition=='gpua100':
            assert self.num_configs <= 4
        elif self.partition=='gpup100':
            assert self.num_configs <= 2

        print(f"{self.num_configs} configs to run.")
            

    def generate_slurm_file(self, verbose=0):
        os.makedirs(self.SLURM_PATH, exist_ok=True)

        print('\n• Generating slurm file...')
        
        text = "#!/bin/bash" + self.SEP
        text += self.SEP

        logfile_output = 'log_' + self.wandb_run_name(array_idx_marker="%a", extension='out')
        logfile_error  = 'log_' + self.wandb_run_name(array_idx_marker="%a", extension='err')

        text += "#SBATCH --job-name=crdb" + self.SEP
        text += f"#SBATCH --output={self.LOGFILES_PATH}/{self.group_name}/" + logfile_output + self.SEP
        text += f"#SBATCH --error={self.LOGFILES_PATH}/{self.group_name}/"  + logfile_error  + self.SEP
        text += self.SEP
        
        if self.cluster==self.CLUSTER_JEAN_ZAY:
            text += f"#SBATCH --qos={self.qos}" + self.SEP
            text += f"#SBATCH --constraint={self.constraint}" + self.SEP
        elif self.cluster==self.CLUSTER_RUCHE:
            text += f"#SBATCH --partition={self.partition}" + self.SEP
        else:
            raise ValueError(f'Cluster {self.cluster} not supported.')
        
        text += f"#SBATCH --time={self.time}" + self.SEP
        text += self.SEP

        text += f"#SBATCH --array=0-{self.num_configs-1}%{self.num_configs}" + self.SEP
        text += "#SBATCH --ntasks=1" + self.SEP
        text += self.SEP
        
        text += "#SBATCH --gres=gpu:1" + self.SEP
        if self.cluster==self.CLUSTER_JEAN_ZAY:
            text += "#SBATCH --cpus-per-task=4" + self.SEP
            text += "#SBATCH --hint=nomultithread" + self.SEP
        text += self.SEP

        if self.cluster==self.CLUSTER_RUCHE:
            text += "#SBATCH --mail-user=jeanvasso@gmail.com" + self.SEP
            text += "#SBATCH --mail-type=ALL" + self.SEP
            text += self.SEP

        text += "module purge" + self.SEP

        if self.cluster==self.CLUSTER_JEAN_ZAY:
            text += "module load anaconda-py3/2023.09" + self.SEP
            text += "conda activate llm4planning" + self.SEP
        elif self.cluster==self.CLUSTER_RUCHE:
            text += "module load anaconda3/2022.10/gcc-11.2.0" + self.SEP
            text += "module load cuda/12.2.1/gcc-11.2.0 " + self.SEP
            text += "source activate llm4planning" + self.SEP
            text += "cd /gpfs/workdir/vassoyanj/repos/llm-planning" + self.SEP
        
        text += "export WANDB__SERVICE_WAIT=300" + self.SEP
        text += "export WANDB_MODE=offline" + self.SEP
        text += "set -x" + self.SEP
        text += "nvidia-smi" + self.SEP
        text += self.SEP

        run_name = self.wandb_run_name(array_idx_marker="${SLURM_ARRAY_TASK_ID}", extension=None)

        arg_config_name = "config_" + run_name
        arg_run_name    = run_name
        arg_group_name  = f"{self.group_name}"
        arguments = arg_config_name + " " + arg_run_name + " " + arg_group_name
        
        if self.cluster==self.CLUSTER_JEAN_ZAY:
            text += "srun python -m src.rl_train " + arguments
        elif self.cluster==self.CLUSTER_RUCHE:
            text += "~/.conda/envs/llm4planning/bin/python -m src.rl_train " + arguments
        else:
            raise ValueError(f'Cluster {self.cluster} not supported.')
        
        self.save_sh_file(text=text, filename=self.slurm_filename, verbose=verbose)
    


    def generate_config_files(self, delete_previous_configs=True, verbose=0):
        if verbose >= 1:
            print(f"\n• Generating {len(self.configs_list)} config files...\n")
        
        os.makedirs(self.CONFIGS_PATH, exist_ok=True)
        if delete_previous_configs: utils.delete_files_in_directory(self.CONFIGS_PATH)

        for idx in range(len(self.configs_list)):
            config_dict = self.configs_list[idx]
            run_name = self.wandb_run_name(array_idx_marker=idx, extension='yaml')
            file_name = 'config_'+run_name
            file_path = os.path.join(self.CONFIGS_PATH, file_name)
            with open(file_path, 'w') as file:
                yaml.dump(config_dict, file)
            
            if verbose >= 1:
                print(f'Generated "{file_path}".')
        
        utils_wandb.initialize_sync_wandb_file(SYNC_WANDB_PATH=self.SYNC_WANDB_PATH)


    def init_log_files(self, verbose=0):
        print('\n• Generating log files...')
        folder_path = os.path.join(self.LOGFILES_PATH, self.group_name)
        os.makedirs(folder_path, exist_ok=True)
        for idx in range(self.num_configs):
            logfile_out = 'log_' + self.wandb_run_name(array_idx_marker=idx, extension='out')
            logfile_err = 'log_' + self.wandb_run_name(array_idx_marker=idx, extension='err')
            filepath_out = os.path.join(folder_path, logfile_out)
            filepath_err = os.path.join(folder_path, logfile_err)
            with open(filepath_out, 'w') as file: pass
            with open(filepath_err, 'w') as file: pass
        print('Done.')
    

    def wandb_run_name(self, array_idx_marker, extension:str=None):
        array_idx_marker = str(array_idx_marker)
        name_no_extension = f"{self.run_name}_" + array_idx_marker
        if extension is None:
            return name_no_extension
        else:
            if not "." in extension: extension = "."+extension
            return name_no_extension + extension


    def save_sh_file(self, text:str, filename:str, verbose=0):
        if not ".sh" in filename: filename += ".sh"
        file_path = os.path.join(self.SLURM_PATH, filename)
        with open(file_path, 'w') as file:
            file.write(text)
        
        if verbose >= 1:
            print(f'Generated "{file_path}".')
    
    
    def print_instructions(self):
        if self.cluster==self.CLUSTER_JEAN_ZAY:
            if self.qos in {self.QOS_DEV, self.QOS_T3, self.QOS_T4} and self.constraint == "v100-32g":
                run_line = f"sbatch -A sur@v100 slurm/{self.slurm_filename}"
            elif self.qos in {self.QOS_DEV, self.QOS_T3, self.QOS_T4} and self.constraint == "a100":
                run_line = f"sbatch -A sur@a100 slurm/{self.slurm_filename}"
            else:
                raise NotImplementedError()
        elif self.cluster==self.CLUSTER_RUCHE:
            run_line = f"sbatch slurm/{self.slurm_filename}"
        else:
            raise ValueError()
        
        print("\nPlease run:")
        print(run_line)
    

    @abstractmethod
    def adjust_config_to_constraints(config:dict, slurm_kwargs:dict, cluster_name:str):
        raise NotImplementedError()
    

    @staticmethod
    def shorten_cluster_name(cluster_name:str):
        if cluster_name == SlurmGenerator.CLUSTER_JEAN_ZAY:
            return 'JZ'
        elif cluster_name == SlurmGenerator.CLUSTER_RUCHE:
            return 'R'
        else:
            raise ValueError(f'Cluster name {cluster_name} not supported.')
        