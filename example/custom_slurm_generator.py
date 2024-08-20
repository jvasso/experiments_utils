from src import SlurmGenerator


class CustomSlurmGenerator(SlurmGenerator):
    def adjust_config_to_constraints(config:dict, slurm_kwargs:dict, cluster_name:str):
        pass