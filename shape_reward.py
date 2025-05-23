import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import random
import os
from observation_parser import ObservationParser

# reward shape: [num_envs, 6]

# Weight order is as follows:
# WinLossRewardFunction(),
# ResourceGatherRewardFunction(),
# ProduceWorkerRewardFunction(),
# ProduceBuildingRewardFunction(),
# AttackRewardFunction(),
# ProduceCombatUnitRewardFunction(),

# reward = afterGs.winner()==maxplayer ? 1.0 : -1.0;
# public static float RESOURCE_RETURN_REWARD = 1;
# public static float RESOURCE_HARVEST_REWARD = 1;
# public static float WORKER_PRODUCE_REWARD = 1;
# public static float BUILDING_PRODUCE_REWARD = 1;
# public static float ATTACK_REWARD = 1;
# public static float COMBAT_UNITS_PRODUCE_REWARD = 1;

class RewardShaper:
    def __init__(self):
        self.reshaped_reward = 0
        
        self.reward_boost = 1.5
        self.decay_timestep = 1000
        self.decay_rate = 0.001
        
        self.defense_penalty_range = 3
        self.defense_penalty = 0.2
        
        self.worker_decay_start = 10
        self.worker_decay_rate = 0.2
        # self.combat_decay_start = 30
        # self.combat_decay_rate = 0.1
        self.building_decay_start = 4
        self.building_decay_rate = 0.2
        
        # self.reward_transfer_start = 
        # self.reward_transfer_rate = 

    def get_reshaped_reward(self, obs_parser: ObservationParser, original_reward, timestep, reward_weight):
        
        original_reward *= self.reward_boost
        
        # Reward Transfer (curriculum)
        transffered_reward = original_reward # [num_envs, 6]
        # if(timestep > self.reward_transfer_start):
        #     transffered_reward[0] = transffered_reward[0]
        #     transffered_reward[1] = transffered_reward[1] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transffered_reward[2] = transffered_reward[2] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transffered_reward[3] = transffered_reward[3] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transffered_reward[4] = transffered_reward[4] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transffered_reward[5] = transffered_reward[5] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        
        # Time Penalty
        time_penalty = [0 for _ in range(obs_parser.num_env)] # [num_envs]
        if(timestep > self.decay_timestep):
            time_penalty = min(transffered_reward, transffered_reward * (self.decay_rate * (timestep - self.decay_timestep)))
            
        # Defense Penalty
        defense_penalty = self.get_denfense_penalty(obs_parser, self.defense_penalty_range) # [num_envs]
        
        # Type Penalty
        type_penalty = self.get_type_penalty(obs_parser, transffered_reward, reward_weight) # [num_envs]
        
        transffered_reward = transffered_reward.sum(axis=1) # [num_envs]
        
        # print(transffered_reward, time_penalty, defense_penalty, type_penalty)
        
        self.reshaped_reward = transffered_reward - time_penalty - defense_penalty - type_penalty
        self.reshaped_reward = [0.0 if x < 0 else x for x in self.reshaped_reward]
        result = [
            r if o < 0 else rr
            for o, rr, r in zip(transffered_reward, self.reshaped_reward, transffered_reward)
        ]
        return result
    
    def get_denfense_penalty(self, obs_parser: ObservationParser, check_range = 3):
        defense_penalty = [0 for _ in range(obs_parser.num_env)]
        def is_valid(h, w):
            return h >= 0 and w >= 0 and h < obs_parser.H and w < obs_parser.W
        
        for e in range(obs_parser.num_env):
            for b_pos in obs_parser.bases_pos[e]: #e_th env's bases positions
                for h in range(b_pos[0] - check_range , b_pos[0] + check_range + 1):
                    for w in range(b_pos[1] - check_range , b_pos[1] + check_range + 1):
                        if not is_valid(h, w):
                            continue
                        grid = obs_parser.parsed_obs[e][h][w]
                        if grid.owner == 1 and grid.unit_types in (4, 5, 6, 7): #check wether it is an enemy's worker, light, heavy or range
                            defense_penalty[e] += self.defense_penalty # TODO, decide the value...
            defense_penalty[e] = min(1, defense_penalty[e])
        
        return defense_penalty
    
    def get_type_penalty(self, obs_parser: ObservationParser, original_reward, reward_weight):
        # production of combat unit 5
        # combat_reward = [0 for _ in range(obs_parser.num_env)]
        # for e in range(obs_parser.num_env):
        #     combat_reward[e] += original_reward[e][5]
        # combat_reward /= reward_weight[5]
        # if('''combat unit count''' > self.combat_decay_start):
        #     combat_reward = combat_reward * (1 - self.combat_decay_rate * ('''combat unit count''' - self.combat_decay_start))
        
        # production of building 3
        building_reward = [0 for _ in range(obs_parser.num_env)]
        for e in range(obs_parser.num_env):
            building_reward[e] += original_reward[e][3]
        building_reward /= reward_weight[3]
        
        building_penalty = [0 for _ in range(obs_parser.num_env)]
        for e in range(obs_parser.num_env):
            building_count = len(obs_parser.bases_pos[e]) + len(obs_parser.barracks_pos[e]) 
            if(building_count > self.building_decay_start):
                building_penalty[e] = min(building_reward[e], (self.building_decay_rate * (building_count - self.building_decay_start)))
            else:
                building_penalty[e] = 0
        
        # production of worker 2
        worker_reward = [0 for _ in range(obs_parser.num_env)]
        for e in range(obs_parser.num_env):
            worker_reward[e] += original_reward[e][2]
        worker_reward /= reward_weight[2]
        
        worker_penalty = [0 for _ in range(obs_parser.num_env)]
        for e in range(obs_parser.num_env):
            worker_count = len(obs_parser.workers_pos[e])
            if(worker_count > self.worker_decay_start):
                worker_penalty[e] = min(worker_reward[e], (self.worker_decay_rate * (worker_count - self.worker_decay_start)))
            else:
                worker_penalty[e] = 0
        
        type_penalty = list(np.array(building_penalty) + np.array(worker_penalty))
        return type_penalty
    