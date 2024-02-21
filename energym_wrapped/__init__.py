from gymnasium.envs.registration import register

register(
    id='MixedUse-v0.1',
    entry_point='energym_wrapped.src.mixeduse.env:WrapperMixedUseEnv',
    max_episode_steps=50000,
)

register(
    id='MixedUse-v0.2',
    entry_point='energym_wrapped.src.mixeduse.window_env:WrapperMixedUseEnv',
    max_episode_steps=50000,
)
