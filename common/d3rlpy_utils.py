import os
import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy import algos
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer, InfiniteBuffer


def collect(env, policy, buffer, n_steps, explore_ratio=0.0, explorer=None):
    """Customized collect method from d3rlpy.algos.qlearning.base
    """
    # start training loop
    episode_idx, rollout_return = 1, 0.
    rollout_returns = []
    observation, _ = env.reset()
    for total_step in range(1, n_steps + 1):
        # sample exploration action
        if np.random.random() > explore_ratio:
            action = policy.predict(np.expand_dims(observation, axis=0))[0]
        else:
            if explorer:
                x = observation.reshape((1,) + observation.shape) # same as np.expand_dims
                action = explorer.sample(policy, x, total_step)[0]
            else:
                action = policy.sample_action(np.expand_dims(observation, axis=0))[0]

        # step environment
        next_observation, reward, terminal, truncated, _ = env.step(action)
        
        # store observation
        buffer.append(observation, action, float(reward))
        rollout_return += reward

        clip_episode = terminal or truncated
        # reset if terminated
        if clip_episode:
            buffer.clip_episode(terminal)
            observation, _ = env.reset()
            # print(f'Episode: {episode_idx}, rollout return: {rollout_return}.')
            rollout_returns.append(round(rollout_return, 2))
            episode_idx += 1
            rollout_return = 0.0
        else:
            observation = next_observation

    # clip the last episode
    buffer.clip_episode(False)

    return buffer, rollout_returns

def collect_dataset_from_trained_policy(env,
                                        model_file,
                                        data_file,
                                        buffer_size,
                                        cache_size,
                                        explore_ratio,
                                        explorer=None,
                                        device='cuda:0',
    ):
    # setup algorithm
    policy = d3rlpy.load_learnable(model_file, device=device)
    
    # customized due to cache_size
    buffer = FIFOBuffer(limit=buffer_size)
    dataset = ReplayBuffer(
        buffer,
        env=env,
        cache_size=cache_size,
    )

    # start data collection
    dataset, rollout_rets = collect(env, policy, dataset, buffer_size, explore_ratio=explore_ratio, explorer=explorer)

    with open(data_file.replace('.h5', '') + '_log.txt', 'w') as fout:
        print(f'Number of transitions: {len(dataset.buffer)}, {dataset.transition_count}', file=fout)
        print(f'Number of episodes: {dataset.size()}', file=fout)
        for ret in rollout_rets:
            print(ret, file=fout)
    
    # save ReplayBuffer
    with open(data_file, "w+b") as f:
        dataset.dump(f)
    return


def collect_dataset_from_training_policy(env,
                                         data_file,
                                         buffer_size,
                                         cache_size,
                                         algo_name,
                                         algo_kwargs,
                                         fit_kwargs,
                                         eval_env=None,
                                         seed=313,
                                         device='cuda:0',
    ):
    online_train_with_buffer(env, buffer_size, cache_size,
                             algo_name, algo_kwargs, fit_kwargs,
                             eval_env, True, data_file, seed, device
                        )


def online_train_with_buffer(env,
                             buffer_size,
                             cache_size,
                             algo_name, 
                             algo_kwargs,
                             fit_kwargs,
                             eval_env=None,
                             needs_save=False,
                             data_file=None,
                             seed=313,
                             device='cuda:0',
    ):
    # set random seeds in random module, numpy module and PyTorch module.
    d3rlpy.seed(seed)

    # setup algorithm
    if algo_name == 'ddpg':
        policy = algos.DDPGConfig(**algo_kwargs).create(device=device)
    elif algo_name == 'sac':
        policy = algos.SACConfig(**algo_kwargs).create(device=device)
    elif algo_name == 'td3':
        policy = algos.TD3Config(**algo_kwargs).create(device=device)
    else:
        raise NotImplementedError('Algorithm is not supported!')

    # setup replay buffer
    # buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)
    # customized due to cache_size
    buffer = FIFOBuffer(limit=buffer_size)
    dataset = ReplayBuffer(
        buffer,
        env=env,
        cache_size=cache_size,
    )

    # prepare exploration strategy if necessary
    explorer = algos.NormalNoise(mean=0.0, std=0.1,)
    if algo_name in ['sac']:
        explorer = None

    # start data collection
    policy.fit_online(env,
                      dataset,
                      explorer=explorer,
                      eval_env=eval_env,
                      **fit_kwargs,
                )

    # save ReplayBuffer
    if needs_save:
        assert data_file is not None
        with open(data_file, "w+b") as f:
            dataset.dump(f)


def offline_train(data_file,
                  algo_name,
                  algo_kwargs,
                  fit_kwargs,
                  eval_env=None,
                  device='cuda:0',
                  seed=128,
    ):
    d3rlpy.seed(seed)

    # load from HDF5
    with open(data_file, "rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())
    print(type(dataset))
    print(f'Number of transitions: {len(dataset.buffer)}, {dataset.transition_count}')
    print(f'Number of episodes: {dataset.size()}')
    print(dataset.dataset_info)
    print(dataset.buffer[0])

    if algo_name == 'ddpg':
        policy = algos.DDPGConfig(**algo_kwargs).create(device=device)
    elif algo_name == 'td3':
        policy = algos.TD3Config(**algo_kwargs).create(device=device)
    elif algo_name == 'sac':
        policy = algos.SACConfig(**algo_kwargs).create(device=device)
    elif algo_name == 'bc':
        policy = algos.BCConfig(**algo_kwargs).create(device=device)
    elif algo_name == 'bcq':
        policy = algos.BCQConfig(**algo_kwargs).create(device=device)
    elif algo_name == 'cql':
        policy = algos.CQLConfig(**algo_kwargs).create(device=device)
    elif algo_name == 'td3+bc':
        policy = algos.TD3PlusBCConfig(**algo_kwargs).create(device=device)
    else:
        raise NotImplementedError('Algorithm is not suppoted!')

    policy.fit(dataset,
               eval_env=eval_env,
               **fit_kwargs
               # epoch_callback=online_callback,
    )


def offline_evaluation(data_file, model_dir, opc_threshold, fast_step=None, device='cuda:0'):
    from glob import glob
    from d3rlpy.metrics import InitialStateValueEstimationEvaluator, SoftOPCEvaluator
    from .utils import sort_model_files
    
    # load dataset and policy
    with open(data_file, "rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())
    print(dataset.dataset_info)

    model_files = glob(os.path.join(model_dir, "*.d3")) # list
    sorted_model_files = sort_model_files(model_files)
    output_file = os.path.join(model_dir, 'offline_evaluation.csv')

    with open(output_file, 'a') as fout:
        for idx, (model_file, training_step) in enumerate(sorted_model_files):
            if fast_step is not None and training_step % fast_step != 0:
                continue
            # print(f'Starting test for {os.path.basename(model_file)}')
            policy = d3rlpy.load_learnable(model_file, device=device)
            # almost same time consumed, 110s for 5e5 dataset
            init_score = InitialStateValueEstimationEvaluator()(policy, dataset)
            opc_score = SoftOPCEvaluator(opc_threshold)(policy, dataset)
            init_score, opc_score = round(init_score, 3), round(opc_score, 3)
            res = [str(idx+1), str(training_step), str(init_score), str(opc_score), str(opc_threshold)]
            print(','.join(res), file=fout)
