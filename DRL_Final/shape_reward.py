import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import random
import os
from DRL_Final.observation_parser import ObservationParser

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

        # Reward Transfer (curriculum)
        transferred_reward = original_reward # [num_envs, 6]
        # if(timestep > self.reward_transfer_start):
        #     transferred_reward[0] = transferred_reward[0]
        #     transferred_reward[1] = transferred_reward[1] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transferred_reward[2] = transferred_reward[2] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transferred_reward[3] = transferred_reward[3] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transferred_reward[4] = transferred_reward[4] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))
        #     transferred_reward[5] = transferred_reward[5] * max(0, 1 - self.reward_transfer_rate * (episode - self.reward_transfer_start))

        # Defense Penalty
        defense_penalty = self.get_denfense_penalty(obs_parser, self.defense_penalty_range) # [num_envs]
        #print(defense_penalty)
        # Type Penalty
        type_penalty = self.get_type_penalty(obs_parser, transferred_reward, reward_weight) # [num_envs]
        #print(type_penalty)

        transferred_reward = np.dot(transferred_reward, reward_weight) # [num_envs]
        
        # Time Penalty
        time_penalty = [0 for _ in range(obs_parser.num_env)] # [num_envs]
        for e in range(obs_parser.num_env):
            if(timestep[e] > self.decay_timestep):
                time_penalty[e] = min(transferred_reward[e], transferred_reward[e] * max(0, (self.decay_rate * (timestep[e] - self.decay_timestep))))

        # print(transferred_reward, time_penalty, defense_penalty, type_penalty)

        reshaped_reward = transferred_reward - time_penalty - defense_penalty - type_penalty
        reshaped_reward = [0.0 if x < 0 else x for x in reshaped_reward]
        result = [
            r if r < 0 else rr
            for r, rr in zip(transferred_reward, reshaped_reward)
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
                        if grid.owner == 2 and grid.unit_types in (4, 5, 6, 7): #check wether it is an enemy's worker, light, heavy or range
                            defense_penalty[e] += self.defense_penalty # TODO, decide the value...
            defense_penalty[e] = min(1, defense_penalty[e])

        return defense_penalty

    def get_type_penalty(self, obs_parser: ObservationParser, original_reward, reward_weight):
        # production of building 3
        building_reward = [0 for _ in range(obs_parser.num_env)]
        for e in range(obs_parser.num_env):
            building_reward[e] += original_reward[e][3]
        building_reward = np.array(building_reward) / reward_weight[3]

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
        #print(worker_reward)
        worker_reward = np.array(worker_reward) / reward_weight[2]

        worker_penalty = [0 for _ in range(obs_parser.num_env)]
        for e in range(obs_parser.num_env):
            worker_count = len(obs_parser.workers_pos[e])
            #print('e:' ,e, '| worker_count:', worker_count)
            if(worker_count > self.worker_decay_start):
                worker_penalty[e] = min(worker_reward[e], (self.worker_decay_rate * (worker_count - self.worker_decay_start)))
            else:
                worker_penalty[e] = 0

        type_penalty = list(np.array(building_penalty) + np.array(worker_penalty))
        return type_penalty