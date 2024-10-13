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
    QOS_DEV_A100 = "qos_gpu_a100-dev"
    QOS_T3  = "qos_gpu-t3"
    QOS_T3_A100 = "qos_gpu_a100-t3"
    QOS_T4  = "qos_gpu-t4"
    CLUSTER_JEAN_ZAY = "jz"
    CLUSTER_RUCHE    = "r"
    
    # mandatory attributes
    PROJECT_PATH    = None
    CONFIGS_PATH    = None
    SLURM_PATH      = None
    LOGFILES_PATH   = None
    SYNC_WANDB_PATH = None
    TRAIN_FILES_FOLDER_PATH = os.path.join('.', 'src')
    CONDA_ENV_NAME          = None
    EMAIL                   = None
    
    # mandatory attributes for RUCHE
    ANACONDA_MODULE_RUCHE = None
    CUDA_MODULE_RUCHE     = None
    CONDA_ENV_PATH_RUCHE  = None
    REPO_PATH_RUCHE       = None

    # mandatory attributes for JEAN-ZAY
    ANACONDA_MODULE_JEAN_ZAY = 'anaconda-py3/2023.09'

    
    def __init__(self,
                 configs_list:List[dict], filename:str, run_name:str, group_name:str,
                 cluster_name:str, time:str, cpu_per_task:int=None, qos:str=None, constraint:str=None, partition:str=None):
        self.cluster      = cluster_name
        self.filename     = filename
        self.configs_list = configs_list
        self.run_name     = run_name
        self.group_name   = group_name
        self.time         = time
        self.cpu_per_task = cpu_per_task
        self.qos          = qos
        self.constraint   = constraint
        self.partition    = partition

        assert not "." in self.filename, f'Please provide filename "{self.filename}" without extension.'
        self.slurm_filename = f'{self.filename}_{self.cluster}.sh'

        assert not ('.sh' in self.SYNC_WANDB_PATH)
        one_config = configs_list[0]
        self.sync_wandb_filepath = self.build_sync_wandb_filepath(script_filename=self.filename, config=one_config)
        
        self.num_configs = len(self.configs_list)

        self.check_class_attr()

        if self.cluster==self.CLUSTER_JEAN_ZAY:
            self.preprocess_jean_zay()
            self.check_jean_zay_attr()
        elif self.cluster==self.CLUSTER_RUCHE:
            self.preprocess_ruche()
            self.check_ruch_attr()
        else:
            raise ValueError(f'Cluster name "{self.cluster}" not supported.')
    

    @classmethod
    def build_sync_wandb_filepath(cls, script_filename:str, config:dict):
        assert cls.SYNC_WANDB_PATH is not None
        assert isinstance(config, dict)
        exp_name = config['exp_name'] if 'exp_name' in config.keys() else 'exp'
        sync_wandb_filename = f'sync_wandb_{script_filename}_{exp_name}.sh'
        filepath = os.path.join(cls.SYNC_WANDB_PATH, sync_wandb_filename)
        return filepath
    

    def check_class_attr(self):
        for class_attr in [self.SLURM_PATH, self.CONFIGS_PATH, self.LOGFILES_PATH, self.SYNC_WANDB_PATH, self.PROJECT_PATH,
                           self.TRAIN_FILES_FOLDER_PATH, self.CONDA_ENV_NAME]:
            assert class_attr is not None
    
    def check_ruch_attr(self):
        for class_attr in [self.ANACONDA_MODULE_RUCHE, self.CUDA_MODULE_RUCHE, self.CONDA_ENV_PATH_RUCHE, self.REPO_PATH_RUCHE]:
            assert class_attr is not None

    def check_jean_zay_attr(self):
        for class_attr in [self.ANACONDA_MODULE_JEAN_ZAY]:
            assert class_attr is not None
    

    def preprocess_jean_zay(self):
        if self.num_configs > 10:
            print(f"\nWARNING: num of configs {self.num_configs} greater than 10.\n")
        else:
            print(f"{self.num_configs} configs to run.")
    

    def preprocess_ruche(self):
        # max num configs = num gpus available per user * 2 or 4 (depending on gpu size)
        # if self.partition in {'gpu','gpu_test'}: # 8*2
        #     assert self.num_configs <= 16
        # elif self.partition=='gpua100':
        #     assert self.num_configs <= 16 # 4*4
        # elif self.partition=='gpup100':
        #     assert self.num_configs <= 4 # 2*2

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
            if not self.constraint=='a100':
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
            assert self.cpu_per_task is not None
            text += f"#SBATCH --cpus-per-task={self.cpu_per_task}" + self.SEP
            text +=  "#SBATCH --hint=nomultithread" + self.SEP
        text += self.SEP

        if self.cluster==self.CLUSTER_RUCHE:
            if self.EMAIL is not None:
                text += f"#SBATCH --mail-user={self.EMAIL}" + self.SEP
                text +=  "#SBATCH --mail-type=ALL" + self.SEP
                text += self.SEP

        text += "module purge" + self.SEP

        if self.cluster==self.CLUSTER_JEAN_ZAY:
            if self.constraint=='a100':
                text += "module load arch/a100" + self.SEP
            text += f"module load {self.ANACONDA_MODULE_JEAN_ZAY}" + self.SEP
            text += f"conda activate {self.CONDA_ENV_NAME}" + self.SEP
        elif self.cluster==self.CLUSTER_RUCHE:
            text += f"module load {self.ANACONDA_MODULE_RUCHE}" + self.SEP
            text += f"module load {self.CUDA_MODULE_RUCHE} " + self.SEP
            text += f"source activate {self.CONDA_ENV_NAME}" + self.SEP
            text += f"cd {self.REPO_PATH_RUCHE}" + self.SEP
        
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

        filename_with_ext = self.filename + ".py" if not ".py" in self.filename else self.filename
        full_filepath = os.path.join(self.TRAIN_FILES_FOLDER_PATH, filename_with_ext)
        assert os.path.isfile(full_filepath), f'File {full_filepath} does not exist.'
        relative_filepath = os.path.relpath(full_filepath, self.PROJECT_PATH)
        dot_path = relative_filepath.replace(os.sep, '.')
        if ".py" in dot_path:
            dot_path = dot_path.split(".py")[0]
        
        if self.cluster==self.CLUSTER_JEAN_ZAY:
            text += f"srun python -m {dot_path} " + arguments
        elif self.cluster==self.CLUSTER_RUCHE:
            python_path = os.path.join(self.CONDA_ENV_PATH_RUCHE, 'bin', 'python')
            text += f"{python_path} -m {dot_path} " + arguments
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
        utils_wandb.initialize_sync_wandb_file(sync_wandb_filepath=self.sync_wandb_filepath)


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
            elif self.qos in {self.QOS_DEV_A100, self.QOS_T3_A100} and self.constraint == "a100":
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
        