import sys
sys.path.append('..')
import gymnasium as gym
from glob import glob

import energym_wrapped
from common.utils import update_algo_kwargs, post_eval_dir
from common.d3rlpy_utils import offline_train


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Arguments')
    parser.add_argument('--algo', type=str, default='cql', help='algorithm name')
    parser.add_argument('--encoder', type=str, default='transformer', help='encoder name')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    args = parser.parse_args()

    eval_env = gym.make("MixedUse-v0.2", weather_list=["GRC_TC_Lamia1"], simulation_days=365, window=20)
    # print(env.get_wrapper_attr('id'))

    algo_kwargs = {'batch_size': args.batch_size}
    algo_kwargs = update_algo_kwargs(algo_kwargs, args.algo, args.encoder, env='mixeduse')

    fit_kwargs = {'n_steps': 10000,
                  'n_steps_per_epoch': 500,
                  'show_progress': False,
    }
    
    data_file = "datasets/MixedUse_ASTL_trained_sac_3e6.h5"

    for idx, seed in enumerate([1234]):
        offline_train(data_file, 'cql', algo_kwargs, fit_kwargs, eval_env=None, seed=seed)
    post_eval_dir(eval_env, 'd3rlpy_logs/CQL_xxxxxxxxxxxxxx', fast=True, eval_step=500)

    eval_env.close()