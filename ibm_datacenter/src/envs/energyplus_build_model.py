# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from re import match
import os

try:
    from src.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Humi import \
        EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Humi
    from src.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi import \
        EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi
except:
    from ibm_datacenter.src.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Humi import \
        EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Humi
    from ibm_datacenter.src.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi import \
        EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi


def build_ep_model(model_file, log_dir, verbose=False, normalized=False, window=0):
    model_basename = os.path.splitext(os.path.basename(model_file))[0]

    if match('2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi(
            model_file=model_file,
            log_dir=log_dir,
            verbose=verbose,
            normalized=normalized,
            window=window,
        )
    # elif match('2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.*', model_basename):
    #     model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose,
    #         normalized=normalized,
    #     )
    elif match('2ZoneDataCenterHVAC_wEconomizer_Temp_Humi.*', model_basename):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Humi(
            model_file=model_file,
            log_dir=log_dir,
            verbose=verbose,
            normalized=normalized,
            window=window,
        )
    # elif match('2ZoneDataCenterHVAC_wEconomizer_Temp.*', model_basename):
    #     model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(
    #         model_file=model_file,
    #         log_dir=log_dir,
    #         verbose=verbose,
    #         normalized=normalized,
    #     )
    else:
        raise ValueError('Unsupported EnergyPlus model')
    return model
