"""
Helpers for computing reward function
"""
import sys
sys.path.append('..')
import numpy as np

from common.utils import temp_penalty


def compute_datacenter_reward_v1(raw_state, temp_weight=1., pw_weight=0.00001):
    try:
        Tenv, Henv, Tz1, Tz2 = raw_state[0:4]
        # PUE removed from obs
        PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power = raw_state[4:]
    except:
        Tenv, Tz1, Tz2 = raw_state[0:3]
        # PUE removed from obs
        PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power = raw_state[3:]
    
    rwd_temp = temp_penalty(Tz1, center=23.5, tolerance=0.5, sharpness=0.5, weight=0.1)
    rwd_temp += temp_penalty(Tz2, center=23.5, tolerance=0.5, sharpness=0.5, weight=0.1)
    rwd_temp *= temp_weight
    rwd_pw = - Whole_Building_Power * pw_weight

    return rwd_temp + rwd_pw, (rwd_temp, rwd_pw)

def compute_datacenter_reward(raw_state, temp_weight=1., pw_weight=0.00001):
    try:
        Tenv, Henv, Tz1, Tz2 = raw_state[0:4]
        # PUE removed from obs
        PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power = raw_state[4:]
    except:
        Tenv, Tz1, Tz2 = raw_state[0:3]
        # PUE removed from obs
        PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power = raw_state[3:]
    
    rwd_temp = temp_penalty(Tz1, center=23.5, tolerance=1., sharpness=0.5, weight=0.1)
    rwd_temp += temp_penalty(Tz2, center=23.5, tolerance=1., sharpness=0.5, weight=0.1)
    rwd_temp *= temp_weight
    rwd_pw = - Whole_Building_Power * pw_weight

    return rwd_temp + rwd_pw, (rwd_temp, rwd_pw)