# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from abc import ABCMeta, abstractmethod
import os, sys, time
import numpy as np
from glob import glob
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import pandas as pd
import json

sys.path.append('..')
from ibm_datacenter.src.env_utils import compute_datacenter_reward
from ibm_datacenter.src.eplus_utils import extract_energyplus_version


class EnergyPlusModel(metaclass=ABCMeta):
    def __init__(self,
                 model_file,
                 log_dir=None,
                 verbose=False,
                 normalized=False,
                 window=0,
        ):
        self.log_dir = log_dir
        self.model_basename = os.path.splitext(os.path.basename(model_file))[0]
        self.energyplus_version = extract_energyplus_version(model_file)
        self.verbose = verbose
        self.normalized = normalized
        self.window = window
        
        self.setup_spaces()
        self.action = 0.5 * (self.action_space.low + self.action_space.high)
        self.action_prev = self.action
        self.raw_state = None # actually obs from EMS, for rewards
        
        self.timestamp_csv = None
        self.sl_episode = None

        # Progress data
        self.num_episodes = 0
        self.num_episodes_last = 0

        self.reward = None
        self.reward_mean = None

    def set_action(self, action):
        self.action_prev = self.action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # In gym env for TRPO/PPO1/PPO2, action distribution is a gaussian with mu = 0, sigma = 1
        # So it must be scaled back into action_space by the environment.
        if self.normalized:
            self.action = self.action_low + (action + 1.) * 0.5 * (self.action_high - self.action_low)
        else:
            self.action = action

    @abstractmethod
    def setup_spaces(self):
        pass
    
    @abstractmethod
    def get_state_from_obs(self, obs):
        # Note that in our co-simulation environment, the state value of the last time step can not be retrived from EnergyPlus process
        # because EMS framework of EnergyPlus does not allow setting EMS calling point after the last timestep is completed.
        pass
    
    @abstractmethod
    def format_state(self, raw_state):
        pass

    def compute_reward(self):
        # raw_state = self.format_raw_state()
        rwd, details = compute_datacenter_reward(self.raw_state)
        return (rwd, details)
    
    # @abstractmethod
    # def format_raw_state(self):
    #     pass

    #--------------------------------------------------
    # Plotting staffs follow
    #--------------------------------------------------
    def plot(self, log_dir='', csv_file='', **kwargs):
        if log_dir != '':
            if not os.path.isdir(log_dir):
                print('energyplus_model.plot: {} is not a directory'.format(log_dir))
                return
            print('energyplus_plot.plot log={}'.format(log_dir))
            self.log_dir = log_dir
            self.show_progress()
        else:
            if not os.path.isfile(csv_file):
                print('energyplus_model.plot: {} is not a file'.format(csv_file))
                return
            print('energyplus_model.plot csv={}'.format(csv_file))
            self.read_episode(csv_file)
            plt.rcdefaults()
            plt.rcParams['font.size'] = 8
            plt.rcParams['lines.linewidth'] = 1.0
            plt.rcParams['legend.loc'] = 'lower right'
            self.fig = plt.figure(1, figsize=(16, 20), dpi=300)
            self.plot_episode(csv_file)
            plt.show()

    # Show convergence
    def show_progress(self):
        self.monitor_file = self.log_dir + "/monitor.csv"

        # Read progress file
        if not self.read_monitor_file():
            print('Progress data is missing')
            sys.exit(1)

        # Initialize graph
        plt.rcdefaults()
        plt.rcParams['font.size'] = 6
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['legend.loc'] = 'lower right'

        self.fig = plt.figure(1, figsize=(16, 10))

        # Show widgets
        axcolor = 'lightgoldenrodyellow'
        self.axprogress = self.fig.add_axes([0.15, 0.10, 0.70, 0.15], facecolor=axcolor)
        self.axslider = self.fig.add_axes([0.15, 0.04, 0.70, 0.02], facecolor=axcolor)
        axfirst = self.fig.add_axes([0.15, 0.01, 0.03, 0.02])
        axlast = self.fig.add_axes([0.82, 0.01, 0.03, 0.02])
        axprev = self.fig.add_axes([0.46, 0.01, 0.03, 0.02])
        axnext = self.fig.add_axes([0.51, 0.01, 0.03, 0.02])

        # Slider is drawn in plot_progress()

        # First/Last button
        self.button_first = Button(axfirst, 'First', color=axcolor, hovercolor='0.975')
        self.button_first.on_clicked(self.first_episode_num)
        self.button_last = Button(axlast, 'Last', color=axcolor, hovercolor='0.975')
        self.button_last.on_clicked(self.last_episode_num)

        # Next/Prev button
        self.button_prev = Button(axprev, 'Prev', color=axcolor, hovercolor='0.975')
        self.button_prev.on_clicked(self.prev_episode_num)
        self.button_next = Button(axnext, 'Next', color=axcolor, hovercolor='0.975')
        self.button_next.on_clicked(self.next_episode_num)

        # Timer
        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self.check_update)
        self.timer.start()

        # Progress data
        self.axprogress.set_xmargin(0)
        self.axprogress.set_xlabel('Episodes')
        self.axprogress.set_ylabel('Reward')
        self.axprogress.grid(True)
        self.plot_progress()

        # Plot latest episode
        self.update_episode(self.num_episodes - 1)

        plt.show()

    def check_update(self):
        if self.read_monitor_file():
            self.plot_progress()

    def plot_progress(self):
        # Redraw all lines
        self.axprogress.lines = []
        self.axprogress.plot(self.reward, color='#1f77b4', label='Reward')
        #self.axprogress.plot(self.reward_mean, color='#ff7f0e', label='Reward (average)')
        self.axprogress.legend()
        # Redraw slider
        if self.sl_episode is None or int(round(self.sl_episode.val)) == self.num_episodes - 2:
            cur_ep = self.num_episodes - 1
        else:
            cur_ep = int(round(self.sl_episode.val))
        self.axslider.clear()
        #self.sl_episode = Slider(self.axslider, 'Episode (0..{})'.format(self.num_episodes - 1), 0, self.num_episodes - 1, valinit=self.num_episodes - 1, valfmt='%6.0f')
        self.sl_episode = Slider(self.axslider, 'Episode (0..{})'.format(self.num_episodes - 1), 0, self.num_episodes - 1, valinit=cur_ep, valfmt='%6.0f')
        self.sl_episode.on_changed(self.set_episode_num)

    def read_monitor_file(self):
        # For the very first call, Wait until monitor.csv is created
        if self.timestamp_csv is None:
            while not os.path.isfile(self.monitor_file):
                time.sleep(1)
            # '-1' is a hack to prevent losing the first set of data
            self.timestamp_csv = os.stat(self.monitor_file).st_mtime - 1

        num_ep = 0
        ts = os.stat(self.monitor_file).st_mtime
        if ts > self.timestamp_csv:
            # Monitor file is updated.
            self.timestamp_csv = ts

            def parse_monitor(mfile):
                firstline = mfile.readline()
                assert firstline.startswith('#')
                metadata = json.loads(firstline[1:])
                assert metadata['env_id'] == "EnergyPlus-v0"
                assert set(metadata.keys()) == {'env_id', 't_start'}, \
                    "Incorrect keys in monitor metadata"
                data = pd.read_csv(mfile, index_col=None)
                assert set(data.keys()) == {'l', 't', 'r'}, \
                    "Incorrect keys in monitor logline"
                return data

            with open(self.monitor_file) as f:
                df = parse_monitor(f)

            self.reward = []
            self.reward_mean = []
            self.episode_dirs = []
            self.num_episodes = 0
            cols = ["r", "l"]
            rew_length = zip(df[cols[0]], df[cols[1]])
            for rew, l in rew_length:
                self.reward.append(float(rew) / l)
                self.reward_mean.append(float(rew) / l)

            episodes_root = self.log_dir + '/output/'
            self.episode_dirs = [
                f"{episodes_root}/{ep}"
                for ep in os.listdir(episodes_root)
                if "episode-" in ep
            ]
            self.num_episodes = len(self.episode_dirs)

            if self.num_episodes > self.num_episodes_last:
                self.num_episodes_last = self.num_episodes
                return True
        else:
            return False

    def update_episode(self, ep):
        self.plot_episode(ep)

    def set_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        self.update_episode(ep)

    def first_episode_num(self, val):
        self.sl_episode.set_val(0)

    def last_episode_num(self, val):
        self.sl_episode.set_val(self.num_episodes - 1)

    def prev_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        if ep > 0:
            ep -= 1
            self.sl_episode.set_val(ep)

    def next_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        if ep < self.num_episodes - 1:
            ep += 1
            self.sl_episode.set_val(ep)

    def get_episode_list(self, log_dir='', csv_file=''):
        if (log_dir != '' and csv_file != '') or (log_dir == '' and csv_file == ''):
            print('Either one of log_dir or csv_file must be specified')
            quit()
        if log_dir != '':
            if not os.path.isdir(log_dir):
                print('energyplus_model.dump: {} is not a directory'.format(log_dir))
                return
            print('energyplus_plot.dump: log={}'.format(log_dir))
            #self.log_dir = log_dir

            # Make a list of all episodes
            # Note: Somethimes csv file is missing in the episode directories
            # We accept gziped csv file also.
            csv_list = glob(log_dir + '/output/episode.*/eplusout.csv') \
                       + glob(log_dir + '/output/episode.*/eplusout.csv.gz')
            self.episode_dirs = list(set([os.path.dirname(i) for i in csv_list]))
            self.episode_dirs.sort()
            self.num_episodes = len(self.episode_dirs)
        else: #csv_file != ''
            self.episode_dirs = [ os.path.dirname(csv_file) ]
            self.num_episodes = len(self.episode_dirs)

    # Model dependent methods
    @abstractmethod
    def read_episode(self, ep): pass

    @abstractmethod
    def plot_episode(self, ep): pass

    @abstractmethod
    def dump_timesteps(self, log_dir='', csv_file='', **kwargs): pass

    @abstractmethod
    def dump_episodes(self, log_dir='', csv_file='', **kwargs): pass
