# HVAC_Offline_RL
We study the problem of data-driven HVAC control methods in buildings based on offline reinforcement learning. More specifically, we investigate the algorithm performance and dataset characteristics (i.e., qualitative and quantitative conditions).

## Project Structure

The structures of our project are as follow:
```
common                                              --common utility functions for two buildings
energym_wrapped
|____src                                            --environment codes
|____main                                           --running codes
|____datasets                                       --dataset location
ibm_datacenter
|____src                                            --environment codes
|____main                                           --running codes
|____datasets                                       --dataset location
```

## Required Packages

```
python            3.8.18
d3rlpy            2.2.0
torch             2.1.0
gymnasium         0.29.1
numpy             1.24.4
energym           0.1
```
The installation instruction of the core `energym` and `rl-testbed-for-energyplus` can be found in https://github.com/bsl546/energym and https://github.com/IBM/rl-testbed-for-energyplus.

Some customization of functions in `d3rlpy` and `energym` can be found in https://github.com/cityujun/d3rlpy and https://github.com/cityujun/energym.

## If you use our code, please cite the paper
```
@article{wang2024experimental,
  title={Experimental evaluation of offline reinforcement learning for HVAC control in buildings},
  author={Wang, Jun and Li, Linyan and Liu, Qi and Yang, Yu},
  journal={arXiv preprint arXiv:2408.07986},
  year={2024}
}
```
