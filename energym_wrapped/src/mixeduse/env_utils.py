import sys
sys.path.append('..')
import numpy as np
from collections import OrderedDict

from energym_wrapped.src.mixeduse.constants import OBSERVATION_SPECS, ACTION_SPECS, ZONE_INDICES
from common.utils import normalize_val, temp_penalty


def convert_obs_to_state_v1(input_tuple):
    """
    state is array of length 8
    containing 3 power-related, 2 external-related, 8 zone indoor temp,
    while separate zone 4, zone 5 (control by AHU1), and average of other 6 (share similar property)
    """
    assert type(input_tuple) == OrderedDict
    ret, ord_temps = [], []
    for ii, (k, v) in enumerate(OBSERVATION_SPECS.items()):
        assert k in input_tuple
        if 'Pw' in k:
            ret.append(input_tuple[k] / 10000.)
        elif k in ['Z02_T', 'Z03_T', 'Z08_T', 'Z09_T', 'Z10_T', 'Z11_T']: # two special zone, others similar
            ord_temps.append(input_tuple[k])
        else:
            ret.append(normalize_val(input_tuple[k], v['lower_bound'], v['upper_bound']))
    assert len(ord_temps) == 6
    ret.append(normalize_val(sum(ord_temps) / 6, 10, 40))
    assert len(ret) == 8
    return np.array(ret)


def convert_action_to_control_v2(action):
    '''v2, indoor setpoints being the same except for zone 4, while control other 4
    '''
    assert type(action) == np.ndarray
    assert action.shape == (5, )
    control = {}
    
    indoor_temp_sp = 16. + (action[0] + 1.) * 0.5 * (26 - 16)
    for k, v in ACTION_SPECS.items():
        if 'Thermostat_sp' in k:
            control[k] = [indoor_temp_sp] if 'Z04' not in k else [18.]
    
    for i in range(2):
        control[f'Bd_T_AHU{i+1}_sp'] = [10. + (action[i+1] + 1.) * 0.5 * (30 - 10)]
        control[f'Bd_Fl_AHU{i+1}_sp'] = [0. + (action[i+3] + 1.) * 0.5 * (1. - 0.)]
    
    control['Bd_Fl_AHU1_sp'][0] *= 10 # maximum of AHU1 fan flow is 10.
    return control


def compute_mixeduse_reward_v1(obs, temp_weight=0.1, pw_weight=0.00002):
    rwd_temp = 0.
    for idx in ZONE_INDICES:
        temp = obs[f'Z{idx}_T']
        if idx == '05':
            rwd_temp += 7 * temp_penalty(temp, center=22., tolerance=1., sharpness=0.5, weight=0.2)
        else:
            rwd_temp += temp_penalty(temp, center=22., tolerance=1., sharpness=0.5, weight=0.2)
    rwd_temp *= temp_weight
    rwd_pw = - obs['Fa_Pw_All'] * pw_weight
    return rwd_temp + rwd_pw, (rwd_temp, rwd_pw)
