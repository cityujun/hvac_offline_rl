from gymnasium.envs.registration import register

register(
    id='DataCenter-v0.1',
    entry_point='ibm_datacenter.src.envs.energyplus_env:EnergyPlusEnv',
    max_episode_steps=60000,
)

register(
    id='DataCenter-v0.2',
    entry_point='ibm_datacenter.src.envs.energyplus_window_env:EnergyPlusEnv',
    max_episode_steps=60000,
)