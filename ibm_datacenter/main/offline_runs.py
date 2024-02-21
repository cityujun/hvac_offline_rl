import os, sys
sys.path.append('..')
import gymnasium as gym
from glob import glob

import ibm_datacenter
from common.utils import update_algo_kwargs, post_eval_dir
from common.d3rlpy_utils import offline_train, offline_evaluation
from d3rlpy import metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Arguments')
    parser.add_argument('--algo', type=str, default='cql', help='algorithm name')
    parser.add_argument('--encoder', type=str, default='transformer', help='encoder name')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    args = parser.parse_args()
    
    ## Environment variables
    os.environ['ENERGYPLUS'] = "/usr/local/EnergyPlus-22-2-0/energyplus"
    idf_dir = "src/idf_files"
    weather_dir = "src/weather_files"
    os.environ['ENERGYPLUS_MODEL'] = os.path.join(idf_dir, "2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi.idf")
    os.environ['ENERGYPLUS_WEATHER'] = os.path.join(weather_dir, "CHN_Hong.Kong.SAR.450070_CityUHK.epw")
    eval_env = gym.make('DataCenter-v0.2', normalized=True, window=30)

    algo_kwargs = {'batch_size': args.batch_size}
    algo_kwargs = update_algo_kwargs(algo_kwargs, args.algo, args.encoder, env='datacenter')

    fit_kwargs = {'n_steps': 10000,
                  'n_steps_per_epoch': 500,
                  'show_progress': False,
                #   'evaluators': {
                #     "init_value": metrics.InitialStateValueEstimationEvaluator(),
                #     "soft_opc": metrics.SoftOPCEvaluator(25000)
                #   }
    }
    
    data_file = "datasets/Temp_Fan_Humi_IL_CA_VA_FL_trained_sac_3e6.h5"
    
    for idx, seed in enumerate([1234]):
        offline_train(data_file, 'cql', algo_kwargs, fit_kwargs, eval_env=None, seed=seed) 
    post_eval_dir(eval_env, 'd3rlpy_logs/CQL_xxxxxxxxxxxxxx', fast=True, eval_step=500) 
    
    eval_env.close()