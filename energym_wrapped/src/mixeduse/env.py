import numpy as np
from gymnasium import Env, spaces
import energym

try:
    from src.mixeduse.constants import ACTION_SPECS, OBSERVATION_SPECS, ZONE_INDICES
    from src.mixeduse.env_utils import (
        convert_obs_to_state_v1,
        convert_action_to_control_v2,
        compute_mixeduse_reward_v1,
    )
except:
    from energym_wrapped.src.mixeduse.constants import ACTION_SPECS, OBSERVATION_SPECS, ZONE_INDICES
    from energym_wrapped.src.mixeduse.env_utils import (
        convert_obs_to_state_v1,
        convert_action_to_control_v2,
        compute_mixeduse_reward_v1,
    )


class WrapperMixedUseEnv(Env):
    # metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, weather_list=["GRC_A_Athens"], simulation_days=10):
        assert len(OBSERVATION_SPECS) == 13 # 3+2+8
        assert len(ACTION_SPECS) == 12
        self.observation_space = spaces.Box(low=np.array([0.] * 8),
                                            high=np.array([5., 5.] + [1.] * 6),
                                            dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([-1.] * 5),
                                       high=np.array([1.] * 5),
                                       dtype=np.float32)
        self.weather_list = weather_list
        self.simulation_days = simulation_days
        self.env = None
        self.render_mode = None
        self.id = 'MixedUse'
        self.episode_idx = -1
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if self.env is not None:
            self.close()
        
        self.episode_idx += 1
        weather_idx = self.episode_idx % len(self.weather_list)
        weather = self.weather_list[weather_idx]
        print('current weather_list[{}] is {}'.format(weather_idx, weather))
        self.env = energym.make("MixedUseFanFCU-v0", weather=weather, simulation_days=self.simulation_days)
        self.env.reset()
        
        obs, done = self.env.step(None)
        assert done == False
        # print(type(obs), len(obs))

        self.last_obs = obs
        self.last_control = None
        return convert_obs_to_state_v1(obs), {}
    
    def step(self, action):
        control = convert_action_to_control_v2(action)
        next_obs, done = self.env.step(control)
        reward, rwd_tuple = compute_mixeduse_reward_v1(next_obs)
        _, hour, _, _ = self.env.get_date()
        if done:
            print('EnergyPlusEnv: (done)')
        self.last_obs = next_obs
        self.last_control = control
        info = {'hour': hour, 'obs': next_obs, 'rwd_details': rwd_tuple, 'control': control}
        return convert_obs_to_state_v1(next_obs), reward, done, False, info # False is truncated

    def render(self):
        assert self.render_mode is None
        return None

    def close(self):
        self.env.close()
