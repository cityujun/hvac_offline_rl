# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import os, sys
import time
import numpy as np
from scipy.special import expit
import pandas as pd
import datetime as dt
from gymnasium import spaces
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons

sys.path.append('..')
from ibm_datacenter.src.envs.energyplus_model import EnergyPlusModel
from common.utils import normalize_val
from common.plot_utils import *


class EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi(EnergyPlusModel):
    def __init__(self,
                 model_file,
                 log_dir,
                 verbose=False,
                 normalized=False,
                 window=0,
        ):
        super(EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan_Humi, self).__init__(model_file, log_dir, verbose, normalized, window)
        self.reward_low_limit = -10000.
        self.axepisode = None
        self.num_axes = 5
        self.text_power_consumption = None
        assert self.energyplus_version == (22, 2, 0)
        self.facility_power_output_var_suffix = "Electricity Demand Rate"
        
    def setup_spaces(self):
        # Bound action temperature
        # for recover original actions
        self.action_low = np.array([10.0, 10., 1.75, 1.75])
        self.action_high = np.array([40.0, 40., 7., 7.])
        
        # for baselines3
        if self.normalized:
            self.action_space = spaces.Box(low=np.array([ -1., -1., -1., -1.]),
                                           high=np.array([ 1., 1., 1., 1.]),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=self.action_low,
                                           high=self.action_high,
                                           dtype=np.float32)
        if self.window == 0:
            self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0., 0.,  0.0,   0.0,  0.0]),
                                                high=np.array([1., 1., 1., 1., 10., 20.0, 20.0, 20.0]),
                                                dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=np.array([[0., 0., 0., 0., 0.,  0.0,   0.0,  0.0]] * self.window),
                                                high=np.array([[1., 1., 1., 1., 10., 20.0, 20.0, 20.0]] * self.window),
                                                dtype=np.float64)

    # Performes mapping from raw_state (retrieved from EnergyPlus process as is) to gym compatible state
    #
    #   state[0] = obs[0]: Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)
    #   state[1] = obs[1]: Environment:Site Outdoor Air Humidity [%](TimeStep)
    #   state[2] = obs[2]: WEST ZONE:Zone Air Temperature [C](TimeStep)
    #   state[3] = obs[3]: EAST ZONE:Zone Air Temperature [C](TimeStep)
    #   state[4] = obs[4]: EMS:Power Utilization Effectiveness [%](TimeStep)
    #   state[5] = obs[5]: Whole Building:Facility Total Electricity Demand Rate [W](Hourly)
    #   state[6] = obs[6]: Whole Building:Facility Total Building Electricity Demand Rate [W](Hourly)
    #   state[7] = obs[7]: Whole Building:Facility Total HVAC Electricity Demand Rate [W](Hourly)
    def get_state_from_obs(self, obs):
        # Need to first handle the case that obs is None
        raw_state = np.zeros(8,) if obs is None else obs 
        self.raw_state = raw_state
        return self.format_state(raw_state)
    
    def format_state(self, raw_state):
        return np.array([ normalize_val(raw_state[0], -20, 50),
                          raw_state[1] / 100 * 1.,
                          normalize_val(raw_state[2], 0, 70),
                          normalize_val(raw_state[3], 0, 70),
                          raw_state[4],
                          raw_state[5] / 1e5,
                          raw_state[6] / 1e5,
                          raw_state[7] / 1e5,
                        ])
        # return np.array([raw_state[0], raw_state[1], raw_state[2], raw_state[4], raw_state[5], raw_state[6]])
    
    #--------------------------------------------------
    # Plotting staffs follow
    #--------------------------------------------------
    def read_episode(self, ep):
        from ibm_datacenter.src.env_utils import compute_datacenter_reward

        if type(ep) is str:
            file_path = ep
        else:
            ep_dir = self.episode_dirs[ep]
            for file in ['eplusout.csv', 'eplusout.csv.gz']:
                file_path = ep_dir + '/' + file
                if os.path.exists(file_path):
                    break
            else:
                print('No CSV or CSV.gz found under {}'.format(ep_dir))
                quit()
        print('read_episode: file={}'.format(file_path))
        df = pd.read_csv(file_path).fillna(method='ffill').fillna(method='bfill')
        self.df = df
        date = df['Date/Time']
        date_time = convert_datetime24(date)

        self.outdoor_temp = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
        self.westzone_temp = df['WEST ZONE:Zone Air Temperature [C](TimeStep)']
        self.eastzone_temp = df['EAST ZONE:Zone Air Temperature [C](TimeStep)']

        self.pue = df['EMS:Power Utilization Effectiveness [](TimeStep)']

        #self.westzone_ite_cpu_electric_power = df['WEST ZONE:Zone ITE CPU Electric Power [W](Hourly)']
        #self.westzone_ite_fan_electric_power = df['WEST ZONE:Zone ITE Fan Electric Power [W](Hourly)']
        #self.westzone_ite_ups_electric_power = df['WEST ZONE:Zone ITE UPS Electric Power [W](Hourly)']

        #WEST ZONE INLET NODE:System Node Temperature [C](TimeStep)
        #WEST ZONE INLET NODE:System Node Mass Flow Rate [kg/s](TimeStep)

        self.westzone_return_air_temp = df['WEST ZONE RETURN AIR NODE:System Node Temperature [C](TimeStep)']
        self.westzone_mixed_air_temp = df['WEST ZONE MIXED AIR NODE:System Node Temperature [C](TimeStep)']
        self.westzone_supply_fan_outlet_temp = df['WEST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_dec_outlet_temp = df['WEST ZONE DEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_dec_outlet_setpoint_temp = df['WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_iec_outlet_temp = df['WEST ZONE IEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_iec_outlet_setpoint_temp = df['WEST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_ccoil_air_outlet_temp = df['WEST ZONE CCOIL AIR OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_ccoil_air_outlet_setpoint_temp = df['WEST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_air_loop_outlet_temp = df['WEST AIR LOOP OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_air_loop_outlet_setpoint_temp = df['WEST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']

        #XX self.eastzone_return_air_temp = df['EAST ZONE RETURN AIR NODE:System Node Temperature [C](TimeStep)']
        #XX self.eastzone_mixed_air_temp = df['EAST ZONE MIXED AIR NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_supply_fan_outlet_temp = df['EAST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_dec_outlet_temp = df['EAST ZONE DEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_dec_outlet_setpoint_temp = df['EAST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_iec_outlet_temp = df['EAST ZONE IEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_iec_outlet_setpoint_temp = df['EAST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_ccoil_air_outlet_temp = df['EAST ZONE CCOIL AIR OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_ccoil_air_outlet_setpoint_temp = df['EAST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_air_loop_outlet_temp = df['EAST AIR LOOP OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_air_loop_outlet_setpoint_temp = df['EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']

        # Electric power
        self.total_building_electric_demand_power = df[f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix} [W](Hourly)']
        self.total_hvac_electric_demand_power = df[f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix} [W](Hourly)']
        self.total_electric_demand_power = df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)']

        # Compute reward list
        self.rewards = []
        self.rewards_gaussian1 = []
        self.rewards_trapezoid1 = []
        self.rewards_gaussian2 = []
        self.rewards_trapezoid2 = []
        self.rewards_power = []
        
        for Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power in zip(
                self.outdoor_temp,
                self.westzone_temp,
                self.eastzone_temp,
                self.pue,
                self.total_electric_demand_power,
                self.total_building_electric_demand_power,
                self.total_hvac_electric_demand_power):
            rew, elem = compute_datacenter_reward([Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power])
            self.rewards.append(rew)
            self.rewards_gaussian1.append(elem[0])
            self.rewards_trapezoid1.append(elem[1])
            self.rewards_gaussian2.append(elem[2])
            self.rewards_trapezoid2.append(elem[3])
            self.rewards_power.append(elem[4])
        
        # Cooling and heating setpoint for ZoneControl:Thermostat
        self.cooling_setpoint = []
        self.heating_setpoint = []
        for dt in date_time:
            self.cooling_setpoint.append(24.0)
            self.heating_setpoint.append(23.0)
        
        (self.x_pos, self.x_labels) = generate_x_pos_x_labels(date)

    def plot_episode(self, ep):
        print('episode {}'.format(ep))
        self.read_episode(ep)

        show_statistics('Reward', self.rewards)
        show_statistics('westzone_temp', self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'])
        show_statistics('eastzone_temp', self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'])
        show_statistics('Power consumption', self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'])
        show_statistics('pue', self.pue)
        show_distrib('westzone_temp distribution', self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'])

        if self.axepisode is None: # Does this really help for performance ?
            self.axepisode = []
            for i in range(self.num_axes):
                if i == 0:
                    ax = self.fig.add_axes([0.05, 1.00 - 0.70 / self.num_axes * (i + 1), 0.90, 0.7 / self.num_axes * 0.85])
                else:
                    ax = self.fig.add_axes([0.05, 1.00 - 0.70 / self.num_axes * (i + 1), 0.90, 0.7 / self.num_axes * 0.85], sharex=self.axepisode[0])
                ax.set_xmargin(0)
                self.axepisode.append(ax)
                ax.set_xticks(self.x_pos)
                ax.set_xticklabels(self.x_labels)
                ax.tick_params(labelbottom='off')
                # ax.grid(True)

        idx = 0
        show_west = True

        if True:
            # Plot and outdoor temperature
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            ax.plot(self.outdoor_temp, 'C0', label='Outdoor temperature')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(-10.0, 40.0)

        if True:
            # Plot zone and outdoor temperature
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            ax.plot(self.westzone_temp, 'C0', label='Westzone temperature')
            ax.plot(self.eastzone_temp, 'C1', label='Eastzone temperature')
            # ax.plot(self.outdoor_temp, 'C2', label='Outdoor temperature')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=1., color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=1., color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(20.0, 28.0)

        if False:
            # Plot return air and sestpoint temperature
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            if show_west:
                ax.plot(self.westzone_return_air_temp, 'C0', label='WEST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(self.westzone_dec_outlet_setpoint_temp, 'C1', label='Westzone DEC outlet setpoint temperature')
            else:
                #ax.plot(self.eastzone_return_air_temp, 'C0', label='EAST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(self.eastzone_dec_outlet_setpoint_temp, 'C1', label='Eastzone DEC outlet setpoint temperature')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if True:
            # Plot return air and sestpoint temperature
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            ax.plot(self.df['WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)'], 'C0', label='WEST ZONE Air Mass Flow Rate')
            ax.plot(self.df['EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)'], 'C1', label='EAST ZONE Air Mass Flow Rate')
            ax.legend()
            ax.set_ylabel('Air volume rate (kg/s)')
            ax.set_ylim(0.0, 10.0)

        if False:
            # Plot west zone, return air, mixed air, supply fan
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            if show_west:
                ax.plot(self.westzone_return_air_temp, 'C0', label='WEST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(self.westzone_mixed_air_temp, 'C1', label='WEST ZONE MIXED AIR NODE:System Node Temperature')
                ax.plot(self.westzone_supply_fan_outlet_temp, 'C2', label='WEST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature')
                ax.plot(self.westzone_dec_outlet_temp, 'C3', label='Westzone DEC outlet temperature')
            else:
                #ax.plot(self.eastzone_return_air_temp, 'C0', label='EAST ZONE RETURN AIR NODE:System Node Temperature')
                #ax.plot(self.eastzone_mixed_air_temp, 'C1', label='EAST ZONE MIXED AIR NODE:System Node Temperature')
                ax.plot(self.eastzone_supply_fan_outlet_temp, 'C2', label='EAST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature')
                ax.plot(self.eastzone_dec_outlet_temp, 'C3', label='Eastzone DEC outlet temperature')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if False:
            # Plot west zone ccoil, air loop
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            if show_west:
                ax.plot(self.westzone_iec_outlet_temp, 'C0', label='Westzone IEC outlet temperature')
                ax.plot(self.westzone_ccoil_air_outlet_temp, 'C1', label='Westzone ccoil air outlet temperature')
                ax.plot(self.westzone_air_loop_outlet_temp, 'C2', label='Westzone air loop outlet temperature')
                ax.plot(self.westzone_dec_outlet_setpoint_temp, label='Westzone DEC outlet setpoint temperature', linewidth=0.5, color='gray')
            else:
                ax.plot(self.eastzone_iec_outlet_temp, 'C0', label='Eastzone IEC outlet temperature')
                ax.plot(self.eastzone_ccoil_air_outlet_temp, 'C1', label='Eastzone ccoil air outlet temperature')
                ax.plot(self.eastzone_air_loop_outlet_temp, 'C2', label='Eastzone air loop outlet temperature')
                ax.plot(self.eastzone_dec_outlet_setpoint_temp, label='Eastzone DEC outlet setpoint temperature', linewidth=0.5, color='gray')
            ax.plot(self.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(self.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if False:
            # Plot calculated reward
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            ax.plot(self.rewards, 'C0', label='Reward')
            ax.plot(self.rewards_gaussian1, 'C1', label='Gaussian1')
            ax.plot(self.rewards_trapezoid1, 'C2', label='Trapezoid1')
            ax.plot(self.rewards_gaussian2, 'C3', label='Gaussian2')
            ax.plot(self.rewards_trapezoid2, 'C4', label='Trapezoid2')
            ax.plot(self.rewards_power, 'C5', label='Power')
            ax.legend()
            ax.set_ylabel('Reward')
            ax.set_ylim(-2.0, 2.0)
        
        if True:
            # Plot reward
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            ax.scatter(np.arange(len(self.rewards)), self.rewards, label='Reward', linewidth=0.5)
            ax.scatter(np.arange(len(self.rewards)), self.rewards_power, label='Power', linewidth=0.5)
            ax.legend()
            ax.set_ylabel('Reward')
            ax.set_ylim(-2.0, 2.0)
        
        if True:
            # Plot calculated reward
            ax = self.axepisode[idx]
            idx += 1
            ax.axes.xaxis.set_visible(False)
            ax.scatter(np.arange(len(self.rewards)), self.rewards_gaussian1, label='Gaussian1')
            ax.scatter(np.arange(len(self.rewards)), self.rewards_gaussian2, label='Gaussian2')
            ax.legend()
            ax.set_ylabel('Reward Gaussian')
            ax.set_ylim(0., 1.0)
        
        if True:
            # Plot calculated reward
            ax = self.axepisode[idx]
            idx += 1
            # ax.axes.xaxis.set_visible(False)
            ax.scatter(np.arange(len(self.rewards)), self.rewards_trapezoid1, label='Trapezoid1')
            ax.scatter(np.arange(len(self.rewards)), self.rewards_trapezoid2, label='Trapezoid2')
            ax.legend()
            ax.set_ylabel('Reward Trapezoid')
            ax.set_ylim(-2.0, 2.0)

        if False:
            # Plot PUE
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            ax.plot(self.pue, 'C0', label='PUE')
            ax.legend()
            ax.set_ylabel('PUE')
            ax.set_ylim(top=2.0, bottom=1.0)

        if False:
            # Plot power consumptions
            ax = self.axepisode[idx]
            idx += 1
            # ax.lines = []
            ax.plot(self.total_electric_demand_power, 'C0', label=f'Whole Building:Facility Total {self.facility_power_output_var_suffix}')
            ax.plot(self.total_building_electric_demand_power, 'C1', label=f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix}')
            ax.plot(self.total_hvac_electric_demand_power, 'C2', label=f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix}')
            ax.legend()
            ax.set_ylabel('Power (W)')
            ax.set_xlabel('Simulation days (MM/DD)')
            ax.tick_params(labelbottom='on')

        # Show average power consumption in text
        if self.text_power_consumption is not None:
            self.text_power_consumption.remove()
        self.text_power_consumption = self.fig.text(0.02,  0.25, 'Whole Power:    {:6,.1f} kW'.format(
            np.average(self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)']) / 1000))
        self.text_power_consumption = self.fig.text(0.02,  0.235, 'Building Power: {:6,.1f} kW'.format(
            np.average(self.df[f'Whole Building:Facility Total Building {self.facility_power_output_var_suffix} [W](Hourly)']) / 1000))
        self.text_power_consumption = self.fig.text(0.02,  0.22, 'HVAC Power:     {:6,.1f} kW'.format(
            np.average(self.df[f'Whole Building:Facility Total HVAC {self.facility_power_output_var_suffix} [W](Hourly)']) / 1000))

    #--------------------------------------------------
    # Dump timesteps
    #--------------------------------------------------
    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        def rolling_mean(data, size, que):
            out = []
            for d in data:
                que.append(d)
                if len(que) > size:
                    que.pop(0)
                out.append(sum(que) / len(que))
            return out
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print('{} episodes'.format(self.num_episodes))
        with open('dump_timesteps.csv', mode='w') as f:
            tot_num_rec = 0
            f.write('Sequence,Episode,Sequence in episode,Reward,tz1,tz2,power,Reward(avg1000)\n')
            que = []
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))
                self.read_episode(ep)
                rewards_avg = rolling_mean(self.rewards, 1000, que)
                ep_num_rec = 0
                for rew, tz1, tz2, pow, rew_avg in zip(
                        self.rewards,
                        self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'],
                        rewards_avg):
                    f.write('{},{},{},{},{},{},{},{}\n'.format(tot_num_rec, ep, ep_num_rec, rew, tz1, tz2, pow, rew_avg))
                    tot_num_rec += 1
                    ep_num_rec += 1

    #--------------------------------------------------
    # Dump episodes
    #--------------------------------------------------
    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print('{} episodes'.format(self.num_episodes))
        with open('dump_episodes.dat', mode='w') as f:
            tot_num_rec = 0
            f.write('#Test Ave1  Min1  Max1 STD1  Ave2  Min2  Max2 STD2   Rew     Power [22,25]1 [22,25]2  Ep\n')
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))
                self.read_episode(ep)
                Temp1 = self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)']
                Temp2 = self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)']
                Ave1, Min1, Max1, STD1 = get_statistics(Temp1)
                Ave2, Min2, Max2, STD2 = get_statistics(Temp2)
                In22_25_1 = np.sum((Temp1 >= 22.0) & (Temp1 <= 25.0)) / len(Temp1)
                In22_25_2 = np.sum((Temp2 >= 22.0) & (Temp2 <= 25.0)) / len(Temp2)
                Rew, _, _, _ = get_statistics(self.rewards)
                Power, _, _, _ = get_statistics(self.df[f'Whole Building:Facility Total {self.facility_power_output_var_suffix} [W](Hourly)'])
                
                f.write('"{}" {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:9.2f} {:8.3%} {:8.3%} {:3d}\n'.format('epw', Ave1, Min1, Max1, STD1, Ave2,  Min2, Max2, STD2, Rew, Power, In22_25_1, In22_25_2, ep))
