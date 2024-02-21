# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import os, subprocess
from glob import glob
import shlex
import shutil
import gzip
import numpy as np
from gymnasium import Env
from gymnasium.utils import seeding

try:
    from src.envs.pipe_io import PipeIo
    from src.envs.energyplus_build_model import build_ep_model
except:
    from ibm_datacenter.src.envs.pipe_io import PipeIo
    from ibm_datacenter.src.envs.energyplus_build_model import build_ep_model


class EnergyPlusEnv(Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self,
                 energyplus_file=None,
                 model_file=None,
                 weather_files=None,
                 log_dir=None,
                 normalized=False, # whether action is normalized
                 verbose=False,
                 render_mode=None,
                 window=0,
        ):
        # Verify path arguments
        if energyplus_file is None:
            energyplus_file = os.getenv('ENERGYPLUS')
        if energyplus_file is None:
            raise ValueError('energyplus_env: FATAL: EnergyPlus executable is not specified. Use environment variable ENERGYPLUS.')
        if model_file is None:
            model_file = os.getenv('ENERGYPLUS_MODEL')
        if model_file is None:
            raise ValueError('energyplus_env: FATAL: EnergyPlus model file is not specified. Use environment variable ENERGYPLUS_MODEL.')
        if weather_files is None:
            weather_files = os.getenv('ENERGYPLUS_WEATHER')
        if weather_files is None:
            raise ValueError('energyplus_env: FATAL: EnergyPlus weather file is not specified. Use environment variable ENERGYPLUS_WEATHER.')
        if log_dir is None:
            log_dir = os.getenv('ENERGYPLUS_LOG')
        if log_dir is None:
            log_dir = 'logs'
        
        # Initialize paths
        self.energyplus_file = energyplus_file
        self.model_file = model_file
        self.weather_files = weather_files.split(',')
        print('weather files: ', len(self.weather_files), self.weather_files)
        self.log_dir = log_dir
        
        # Create an EnergyPlus model
        self.ep_model = build_ep_model(
                        model_file=self.model_file,
                        log_dir=self.log_dir,
                        verbose=verbose,
                        normalized=normalized,
                        window=window,
                    )
        self.action_space = self.ep_model.action_space
        self.observation_space = self.ep_model.observation_space
        # TODO: self.reward_space which defaults to [-inf,+inf]
        # print(self.action_space, self.observation_space)

        self.pipe_io = PipeIo()
        self.normalized = normalized
        self.episode_idx = -1
        self.verbose = verbose
        assert render_mode is None #or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.energyplus_process = None
        self.id = 'DataCenter'
        self.window = window
        self.obs_buffer = np.zeros_like(self.observation_space.low)
        print(self.obs_buffer.shape)
        
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.stop_instance()

        self.episode_idx += 1
        self.start_instance()
        self.timestep = 0
        return self.step(None)[0], {}

    def start_instance(self):
        print('Starting new environment')
        assert self.energyplus_process is None

        output_dir = self.get_output_dir()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.pipe_io.start()
        print('start_instance(): idx={}, model_file={}'.format(self.episode_idx, self.model_file))
        
        # Handling of multiple weather files
        weather_idx = self.episode_idx % len(self.weather_files)
        weather_file = self.weather_files[weather_idx]
        print('start_instance(): weather_files[{}]={}'.format(weather_idx, weather_file))

        # Make copies of model file and weather file into output dir, and use it for execution
        # This allow update of these files without affecting active simulation instances
        shutil.copy(self.model_file, output_dir)
        shutil.copy(weather_file, output_dir)
        # copy_model_file = output_dir + '/' + os.path.basename(self.model_file)
        # copy_weather_file = output_dir + '/' + os.path.basename(weather_file)

        # Spawn a process
        cmd = self.energyplus_file \
              + ' -r ' \
              + ' -w ' + os.path.basename(weather_file) \
              + ' ' + os.path.basename(self.model_file)
        print('Starting EnergyPlus with command: %s' % cmd)
        self.energyplus_process = subprocess.Popen(
            shlex.split(cmd),
            shell=False,
            cwd=output_dir
        )

    def stop_instance(self):
        if self.energyplus_process is not None:
            self.energyplus_process.terminate()
            self.energyplus_process = None
        if self.pipe_io is not None:
            self.pipe_io.stop()
        if self.episode_idx >= 0:
            def count_severe_errors(file):
                if not os.path.isfile(file):
                    return -1 # Error count is unknown
                # Sample: '   ************* EnergyPlus Completed Successfully-- 6214 Warning; 2 Severe Errors; Elapsed Time=00hr 00min  7.19sec'
                fd = open(file)
                lines = fd.readlines()
                fd.close()
                for line in lines:
                    if line.find('************* EnergyPlus Completed Successfully') >= 0:
                        tokens = line.split()
                        return int(tokens[6])
                return -1
            epsode_dir = self.get_output_dir()
            file_csv = epsode_dir + '/eplusout.csv'
            file_csv_gz = epsode_dir + '/eplusout.csv.gz'
            file_err = epsode_dir + '/eplusout.err'
            files_to_preserve = ['eplusout.csv', 'eplusout.err', 'eplustbl.htm']
            files_to_clean = ['eplusmtr.csv', 'eplusout.audit', 'eplusout.bnd',
                              'eplusout.dxf', 'eplusout.eio', 'eplusout.edd',
                              'eplusout.end', 'eplusout.eso', 'eplusout.mdd',
                              'eplusout.mtd', 'eplusout.mtr', 'eplusout.rdd',
                              'eplusout.rvaudit', 'eplusout.shd', 'eplusssz.csv',
                              'epluszsz.csv', 'sqlite.err']

            # Check for any severe error
            nerr = count_severe_errors(file_err)
            if nerr != 0:
                print('EnergyPlusEnv: Severe error(s) occurred. Error count: {}'.format(nerr))
                print('EnergyPlusEnv: Check contents of {}'.format(file_err))
                #sys.exit(1)

            # Compress csv file and remove unnecessary files
            # If csv file is not present in some reason, preserve all other files for inspection
            if os.path.isfile(file_csv):
                with open(file_csv, 'rb') as f_in:
                    with gzip.open(file_csv_gz, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(file_csv)

                if not os.path.exists("/tmp/verbose"):
                    for file in files_to_clean:
                        file_path = epsode_dir + '/' + file
                        if os.path.isfile(file_path):
                            os.remove(file_path)

    def step(self, action):
        self.timestep += 1
        # Send action to the environment
        if action is not None:
            # # baselines 0.1.6 changed action type
            assert isinstance(action, np.ndarray) and not isinstance(action[0], np.ndarray)
            self.ep_model.set_action(action)

            if not self.send_action():
                print('EnergyPlusEnv.step(): Failed to send an action. Quitting.')
                observation = (self.observation_space.low + self.observation_space.high) * 0.5
                reward = 0.0
                done = True
                print('EnergyPlusEnv: (quit)')
                return observation, reward, done, False, {}
        
        # Receive observation from the environment
        obs, done = self.receive_observation() # obs will be None for calling at total_timestep + 1
        next_state = self.ep_model.get_state_from_obs(obs)
        reward, reward_details = self.ep_model.compute_reward()

        if done:
            print('EnergyPlusEnv: (done)')
        self.obs_buffer = np.vstack((self.obs_buffer[1:], next_state))
        return self.obs_buffer, reward, done, False, {'rwd_details': reward_details} # False is truncated
    
    def send_action(self):
        num_data = len(self.ep_model.action)
        if self.pipe_io.writeline('{0:d}'.format(num_data)):
            return False
        for i in range(num_data):
            self.pipe_io.writeline('{0:f}'.format(self.ep_model.action[i]))
        self.pipe_io.flush()
        return True

    def receive_observation(self):
        line = self.pipe_io.readline()
        if line == '':
            # This is the (usual) case when we send action data after all simulation timestep have finished.
            return None, True
        num_data = int(line)
        # Number of data received may not be same as the size of observation_space
        # assert num_data == len(self.observation_space.low)
        obs = np.zeros(num_data)
        for i in range(num_data):
            line = self.pipe_io.readline()
            if line == '':
                # This is usually system error
                return None, True
            val = float(line)
            obs[i] = val
        return obs, False
    
    def render(self):
        assert self.render_mode is None
        return None
        
    def close(self):
        self.stop_instance()

    def plot(self, log_dir='', csv_file=''):
        self.ep_model.plot(log_dir=log_dir, csv_file=csv_file)

    def dump_timesteps(self, log_dir='', csv_file='', reward_file=''):
        self.ep_model.dump_timesteps(log_dir=log_dir, csv_file=csv_file)

    def dump_episodes(self, log_dir='', csv_file='', reward_file=''):
        self.ep_model.dump_episodes(log_dir=log_dir, csv_file=csv_file)

    def get_output_dir(self):
        return self.log_dir + '/output/episode-{:08}-{:05}'.format(self.episode_idx, os.getpid())


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    import os

    os.environ['ENERGYPLUS'] = "/usr/local/EnergyPlus-22-2-0/energyplus"
    idf_dir = "src/idf_files"
    weather_dir = "src/weather_files"
    # os.environ['ENERGYPLUS_MODEL'] = os.path.join(idf_dir, "2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi.idf")
    os.environ['ENERGYPLUS_MODEL'] = os.path.join(idf_dir, "2ZoneDataCenterHVAC_wEconomizer_Temp_Humi.idf")
    
    # os.environ['ENERGYPLUS_WEATHER'] = os.path.join(weather_dir, "CHN_Hong.Kong.SAR.450070_CityUHK.epw")
    os.environ['ENERGYPLUS_WEATHER'] = ','.join(glob(os.path.join(weather_dir, "USA_*.epw")))

    env = EnergyPlusEnv(normalized=True, window=15)
    
    # os.environ['ENERGYPLUS_WEATHER'] = os.path.join(weather_dir, "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw")
    # eval_env = EnergyPlusEnv()

    state, _ = env.reset()
    print(state, state.shape)

    # check_env(env)
    env.close()
    # eval_env.close()