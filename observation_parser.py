from collections import namedtuple
import numpy as np

grid_data = namedtuple("grid_data", "hit_points, resources, owner, unit_types, current_action")

class ObservationParser:
    unit_type_dict = {0: "None", 1: "resource", 2: "base", 3: "barrack", 4: "worker", 5: "light", 6: "heavy", 7: "ranged"}
    action_dict = {0: "None", 1: "move", 2: "harvest", 3: "return", 4: "produce", 5: "attack"}

    def __init__(self):
        self.prev_parsed_obs = []
        self.parsed_obs = [] #for each env, parse h*w grid_data

        self.workers = []
        self.lights = []
        self.heavies = []
        self.ranges = []
        self.barracks = []

        self.bases = []
        self.resource_points = []

        self.workers_pos = []
        self.lights_pos = []
        self.heavies_pos = []
        self.ranges_pos = []
        self.barracks_pos = []

        self.bases_pos = []
        self.resource_points_pos = []

        self.enemy_workers_pos = []
        self.enemy_lights_pos = []
        self.enemy_heavies_pos = []
        self.enemy_ranges_pos = []
        self.enemy_barracks_pos = []

        self.enemy_bases_pos = []

        # initialize
        #self.prev_paresed_obs = self.parse(vec_obs)

    def initialize_observation(self, vec_obs):
        self.vec_obs = vec_obs # shape: (num_envs, h, w, 27)
        self.num_env, self.H, self.W, self.C = vec_obs.shape
        self.parse(vec_obs)

    def parse_feature(self, feature):
        return grid_data(np.argmax(feature[0:5]), np.argmax(feature[5:10]), np.argmax(feature[10:13]), np.argmax(feature[13:21]), np.argmax(feature[21:27]))

    def parse(self, vec_obs): #parse 
        self.prev_paresed_obs = self.parsed_obs
        self.vec_obs = vec_obs
        
        #reset
        self.parsed_obs = []
        
        self.workers_pos = [[] for _ in range(self.num_env)]
        self.lights_pos = [[] for _ in range(self.num_env)]
        self.heavies_pos = [[] for _ in range(self.num_env)]
        self.ranges_pos = [[] for _ in range(self.num_env)]
        self.barracks_pos = [[] for _ in range(self.num_env)]

        self.bases_pos = [[] for _ in range(self.num_env)]
        self.resource_points_pos = [[] for _ in range(self.num_env)]

        self.enemy_workers_pos = [[] for _ in range(self.num_env)]
        self.enemy_lights_pos = [[] for _ in range(self.num_env)]
        self.enemy_heavies_pos = [[] for _ in range(self.num_env)]
        self.enemy_ranges_pos = [[] for _ in range(self.num_env)]

        self.enemy_barracks_pos = [[] for _ in range(self.num_env)]
        self.enemy_bases_pos = [[] for _ in range(self.num_env)]

        #parse
        for e in range(self.num_env):
            plane_data = [[] for _ in range(self.H)]
            for i in range(self.H):
                for j in range(self.W):
                    plane_data[i].append(self.parse_feature(vec_obs[e][i][j]))
            self.parsed_obs.append(plane_data)
        return self.parsed_obs
    
    def count_units(self):
        for e in range(self.num_env):
            for h in range(self.H):
                for w in range(self.W):
                    if self.obs[e][h][w].unit_type == 2 and self.obs[e][h][w].owner == 0:
                        self.bases_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 3 and self.obs[e][h][w].owner == 0:
                        self.barracks_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 4 and self.obs[e][h][w].owner == 0:
                        self.workers_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 5 and self.obs[e][h][w].owner == 0:
                        self.lights_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 6 and self.obs[e][h][w].owner == 0:
                        self.heavies_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 7 and self.obs[e][h][w].owner == 0:
                        self.ranges_pos.append((h, w))
                    
                    elif self.obs[e][h][w].unit_type == 2 and self.obs[e][h][w].owner == 1:
                        self.enemy_bases_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 3 and self.obs[e][h][w].owner == 1:
                        self.enemy_barracks_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 4 and self.obs[e][h][w].owner == 1:
                        self.enemy_workers_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 5 and self.obs[e][h][w].owner == 1:
                        self.enemy_lights_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 6 and self.obs[e][h][w].owner == 1:
                        self.enemy_heavies_pos.append((h, w))
                    elif self.obs[e][h][w].unit_type == 7 and self.obs[e][h][w].owner == 1:
                        self.enemy_ranges_pos.append((h, w))
                    
                    
                    
