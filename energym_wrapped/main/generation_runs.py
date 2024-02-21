import os, sys
sys.path.append('..')
import numpy as np
from glob import glob
import gymnasium as gym
from d3rlpy import algos

import energym_wrapped
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

    weather_list = ['GRC_A_Athens', 'GRC_TC_SkiathosAP', 'GRC_TC_Trikala', 'GRC_TC_LarisaAP1']
    # env = gym.make("MixedUse-v0.1", weather_list=["GRC_A_Athens"], simulation_days=365)
    # eval_env = gym.make("MixedUse-v0.1", weather_list=["GRC_A_Athens"], simulation_days=365)
    env = gym.make("MixedUse-v0.2", weather_list=weather_list, simulation_days=365, window=20)
    eval_env = gym.make("MixedUse-v0.2", weather_list=["GRC_TC_Lamia1"], simulation_days=365, window=20)

    """
    Scenario 1: Final Buffer
    """
    data_size = 30 * 100000
    algo_kwargs = {'batch_size': 1024 * 4}
    algo_kwargs = update_algo_kwargs(algo_kwargs, 'sac', args.encoder, env='mixeduse')
    
    fit_kwargs = {'n_steps': data_size,
                  'n_steps_per_epoch': 100000, # 5
                  'update_interval': 100, # 1000 step per epoch
                  'update_start_step': 50000, # 5
                  'random_steps': 50000, # 5
                  'show_progress': False,
    }

    data_file = "datasets/MixedUse_ASTL_training_sac_3e6.h5"
    collect_dataset_from_training_policy(env,
                                         data_file,
                                         buffer_size=int(data_size),
                                         cache_size=int(4e4),
                                         algo_name='sac',
                                         algo_kwargs=algo_kwargs,
                                         fit_kwargs=fit_kwargs,
                                         eval_env=eval_env,
                                )

    """
    Dataset 2: Trained
    """
    model_file = 'd3rlpy_logs/SAC_online/model_3000000.d3'
    data_file = "datasets/MixedUse_ASTL_trained_sac_3e6.h5"
    collect_dataset_from_trained_policy(env,
                                        model_file,
                                        data_file,
                                        buffer_size=int(3e6),
                                        cache_size=int(4e4),
                                        explore_ratio=0.2,
                                        explorer=algos.NormalNoise(mean=0.0, std=0.2),
                            )
    
    env.close()
    eval_env.close()
