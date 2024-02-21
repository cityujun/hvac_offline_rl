import os, sys
sys.path.append('..')
import numpy as np
from glob import glob
import gymnasium as gym
from d3rlpy import algos

import ibm_datacenter
from main.test import online_test
from common.d3rlpy_utils import (
    collect_dataset_from_training_policy,
    collect_dataset_from_trained_policy,
)
from common.utils import update_algo_kwargs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Arguments')
    parser.add_argument('--encoder', type=str, default='transformer', help='encoder name')
    args = parser.parse_args()

    ## Environment variables
    os.environ['ENERGYPLUS'] = "/usr/local/EnergyPlus-22-2-0/energyplus"
    idf_dir = "src/idf_files"
    weather_dir = "src/weather_files"
    os.environ['ENERGYPLUS_MODEL'] = os.path.join(idf_dir, "2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi.idf")
    os.environ['ENERGYPLUS_WEATHER'] = ','.join(glob(os.path.join(weather_dir, "USA_*.epw")))
    env = gym.make('DataCenter-v0.2', normalized=True, window=30)

    os.environ['ENERGYPLUS_WEATHER'] = os.path.join(weather_dir, "CHN_Hong.Kong.SAR.450070_CityUHK.epw")
    eval_env = gym.make('DataCenter-v0.2', normalized=True, window=30)

    """
    Scenario 1: Final Buffer
    """
    data_size = 30 * 100000
    algo_kwargs = {'batch_size': 1024 * 4}
    algo_kwargs = update_algo_kwargs(algo_kwargs, 'sac', args.encoder, env='datacenter')

    fit_kwargs = {'n_steps': data_size,
                  'n_steps_per_epoch': 100000, # 5
                  'update_interval': 100, # 1000 step per epoch
                  'update_start_step': 50000, # 5
                  'random_steps': 50000, # 5
                  'show_progress': False,
    }
    
    data_file = "datasets/Temp_Fan_Humi_IL_CA_VA_FL_training_sac_3e6.h5"
    collect_dataset_from_training_policy(env,
                                        data_file,
                                        buffer_size=int(data_size),
                                        cache_size=int(6e4),
                                        algo_name='sac',
                                        algo_kwargs=algo_kwargs,
                                        fit_kwargs=fit_kwargs,
                                        eval_env=eval_env,
                                        seed=234,
                                    )

    """
    Scenario 2: Trained
    """
    model_file = 'd3rlpy_logs/SAC_online/model_3000000.d3'

    data_file = "datasets/Temp_Fan_Humi_IL_CA_VA_FL_trained_sac_3e6.h5"
    collect_dataset_from_trained_policy(env,
                                        model_file,
                                        data_file,
                                        buffer_size=int(3e6),
                                        cache_size=int(6e4),
                                        explore_ratio=0.2,
                                        explorer=algos.NormalNoise(mean=0.0, std=0.1),
                            )
    
    env.close()
    eval_env.close()
