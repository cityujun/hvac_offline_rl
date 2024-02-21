import os
import numpy as np

from .d3rlpy_encoders import TransformerEncoderFactory
from d3rlpy.models.encoders import DefaultEncoderFactory, register_encoder_factory
register_encoder_factory(TransformerEncoderFactory)


def normalize_val(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)
    ## for ibm-datacenter
    # the range of outdoor temperature is -20 ~ 50 C
    # the range of indoor temperature is 0 ~ 70 C

def recover_val(val, min_val, max_val):
    return (max_val - min_val) * val + min_val


def temp_penalty(val, center=22.5, tolerance=0.5, sharpness=0.5, weight=0.1):
    gaussian = np.exp(- (val - center) ** 2 * sharpness)
    low, high = center - tolerance, center + tolerance
    trapezoid = - max(0., low - val) - max(0., val - high)
    return gaussian + weight * trapezoid


def check_eplus_env(env_name):
    from stable_baselines3.common.env_checker import check_env
    import gymnasium as gym
    
    if env_name == 'mixeduse':
        import energym_wrapped
        env = gym.make("MixedUse-v0.1", weather="GRC_A_Athens", simulation_days=3)
    elif env_name == 'datacenter':
        import ibm_datacenter
        env = gym.make('DataCenter-v0.1', normalized=True)
    else:
        raise NotImplementedError('Env is not supported!')

    check_env(env)
    env.close()


def sort_model_files(model_files):
    file2step = dict()
    for model_file in model_files:
        step = os.path.basename(model_file).replace('model_', '').replace('.d3', '')
        file2step[model_file] = int(step)
    return sorted(file2step.items(), key=lambda x: x[1])


def post_eval_dir(env, model_dir, device='cuda:0', fast=False, eval_step=1000):
    from glob import glob

    model_files = glob(os.path.join(model_dir, "*.d3")) # list
    sorted_model_files = sort_model_files(model_files)
    output_file = os.path.join(model_dir, 'online_evaluation.csv')

    with open(output_file, 'a') as fout:
        for idx, (model_file, training_step) in enumerate(sorted_model_files):
            if fast and training_step % eval_step != 0:
                continue
            # print(f'Starting test for {os.path.basename(model_file)}')
            episode_step, episode_reward, episode_power = evaluate_with_energyplus(env, model_file, device)
            res = [str(idx+1), str(training_step), str(episode_reward), str(episode_power), str(episode_step)]
            print(','.join(res), file=fout)


def evaluate_with_energyplus(env, model_file, device):
    import d3rlpy

    policy = d3rlpy.load_learnable(model_file, device=device)
    env_id = env.get_wrapper_attr('id')

    observation, _ = env.reset()
    episode_reward = 0.0
    episode_powers = []

    while True:
        action = policy.predict(np.expand_dims(observation, axis=0))[0]
        observation, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            break
        
        episode_reward += float(reward)
        if env_id == 'DataCenter':
            if len(observation.shape) == 1:
                episode_powers.append(float(observation[-3] * 100))
            elif len(observation.shape) == 2:
                episode_powers.append(float(observation[-1, -3] * 100))
        elif env_id == 'MixedUse':
            episode_powers.append(float(info['obs']['Fa_Pw_All'] / 1000))
    
    return len(episode_powers), round(episode_reward, 3), round(float(np.mean(episode_powers)), 3)


def generate_encoder(encoder, env):
    if encoder == 'transformer':
        time_len = 30 if env == 'datacenter'  else 20
        return TransformerEncoderFactory(time_len=time_len)
    else:
        assert encoder == 'default'
        return DefaultEncoderFactory()


def update_algo_kwargs(kwargs, algo, encoder, env='datacenter'):
    assert algo in ['bc', 'ddpg', 'sac', 'td3', 'td3+bc', 'bcq', 'cql']
    assert encoder in ['default', 'transformer']
    if algo == 'bc':
        kwargs.update({
            'encoder_factory': generate_encoder(encoder, env),
        })
    elif algo in ['ddpg', 'sac', 'td3', 'td3+bc', 'cql']:
        kwargs.update({
            'actor_encoder_factory': generate_encoder(encoder, env),
            'critic_encoder_factory': generate_encoder(encoder, env),
        })
        if algo == 'td3+bc':
            kwargs.update({
                'alpha': 1.,
                'actor_learning_rate': 0.0003,
                'critic_learning_rate': 0.0003,
            }) # alpha might need to be smaller, default 2.5
    elif algo == 'bcq':
        kwargs.update({
            'actor_encoder_factory': generate_encoder(encoder, env),
            'critic_encoder_factory': generate_encoder(encoder, env),
            'imitator_encoder_factory': generate_encoder(encoder, env),
            'actor_learning_rate': 0.0003,
            'critic_learning_rate': 0.0003,
            'imitator_learning_rate': 0.0003,
        })
    return kwargs
