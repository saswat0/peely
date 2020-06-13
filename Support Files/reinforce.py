import gym
import yaml
import argparse
import torch as T
import numpy as np

from time import sleep

from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

class Policy(nn.Module):
    def __init__(self, layers):
        super(Policy, self).__init__()
        network = []
        
        for idx in range(len(layers)-1):
            network += [nn.Linear(layers[idx], layers[idx+1])]
            network += [nn.ReLU() if idx+2 < len(layers) else nn.Softmax(dim=-1)]
       
        self.network = nn.Sequential(*network)
    
    def forward(self, state):
        return self.network(state)


class REINFORCE:
    def __init__(self, config, device="cuda:0"):
        self.writer = SummaryWriter() 
        self.run_title = config["run-title"]
        self.device = device
        self.env = gym.make(config["environment"])
        self.policy = Policy([
            np.prod(self.env.observation_space.shape), 
            *config["architecture"], 
            self.env.action_space.n
        ]).to(device)
        self.optim = T.optim.Adam(self.policy.parameters(), lr=config["lr"], weight_decay=config["weight-decay"])
        print(self.policy)

    def run_episode(self, state, max_iter = 500):
        log_policy, rewards = [], []
        for _ in range(max_iter):
            state = T.Tensor(state).view(-1).to(self.device)
            action_distribution = self.policy(state)
            action = int(T.distributions.Categorical(action_distribution).sample())
            state, reward, done, _ = self.env.step(action)
            rewards += [reward]
            log_policy += [T.log(action_distribution[action])]
            if done:
                break
        return log_policy, rewards

    def discount_rewards(self, rewards, gamma):
        r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()

    def train(self, epochs, episodes, gamma=0.99, render=False):
        self.policy.train()
        avg_reward = 0 
        for epoch in range(epochs):
            objective = 0
            for episode in range(episodes):
                state = self.env.reset()
                log_policy, rewards = self.run_episode(state)

                self.writer.add_scalar(f'{self.run_title}/reward', sum(rewards), epoch*episodes + episode)
                
                # Cumaltive moving average
                avg_reward += (sum(rewards) - avg_reward) / (epoch*episodes + episode + 1)
                discount_reward = self.discount_rewards(rewards, gamma)
                objective += sum(log_policy*discount_reward)

            loss = -objective.mean()  # invert to represent cost rather than reward

            print('EPOCH:', epoch, f'AVG REWARD: {avg_reward:.2f} | LOSS: {loss:.2f}', end='\r')
            self.writer.add_scalar(f'{self.run_title}/avg_reward', avg_reward, epoch)
            self.writer.add_scalar(f'{self.run_title}/loss', loss, epoch)

            # Update Policy
            self.optim.zero_grad()   
            loss.backward()   
            self.optim.step()   

            # Visualize Episode
            if render and epoch > 20:
                self.visualize()
 
        self.env.close()
        self.writer.close() 


    def visualize(self):
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            state = T.Tensor(state).view(-1).to(self.device)
            action_distribution = self.policy(state)
            action = T.distributions.Categorical(action_distribution).sample()
            state, _, done, _ = self.env.step(int(action))
            sleep(0.01) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="configs/reinforce.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    agent = REINFORCE(config)
    agent.train(epochs=config["epochs"], episodes=config["episodes"], gamma=config["gamma"], render=config["render"])
