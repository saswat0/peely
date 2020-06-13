import os 
import gym
import yaml
import random 
import argparse
import torch as T
import numpy as np

import imageio 

from torch import nn
from torch.nn import functional as F

class Q_Network(nn.Module):
    def __init__(self, layers):
        super(Q_Network, self).__init__()
        network = []
        
        for idx in range(len(layers)-1):
            network += [nn.Linear(layers[idx], layers[idx+1])]
            if idx+2 < len(layers):
                network += [nn.ReLU()]
       
        self.network = nn.Sequential(*network)
    
    def forward(self, state):
        return self.network(state)

class DQN:
    def __init__(self, config):
        self.device = config["device"]
        self.env = gym.make(config["environment"])

        self.num_states  = np.prod(self.env.observation_space.shape)
        self.num_actions = self.env.action_space.n

        layers = [
            self.num_states, 
            *config["architecture"], 
            self.num_actions
        ]

        self.policy_net = Q_Network(layers).to(self.device)
        self.save_path = config["save-path"]

    def get_action(self, state, epsilon):
        """
            Get an action using epsilon-greedy
        """
        if np.random.sample() < epsilon:
            return int(np.random.choice(np.arange(self.num_actions)))
        else:
            return self.policy_net(T.tensor(state, device=self.device).float()).argmax().item()

    def visualize(self, load_path):
        done = False
        state = self.env.reset()

        self.policy_net.load_state_dict(T.load(load_path, map_location=self.device))
        self.policy_net.eval()
        
        images = []
        while not done:
            images += [self.env.render(mode="rgb_array")]
            action = self.get_action(state, -1)
            state, _, done, _ = self.env.step(int(action))

        imageio.mimsave(self.save_path, images)
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="../configs/dqn.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    agent = DQN(config)
    agent.visualize(config["load-path"])