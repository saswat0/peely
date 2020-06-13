import os 
import gym
import yaml
import random 
import argparse
import torch as T
import numpy as np

from time import sleep
from tqdm import tqdm
from collections import namedtuple

from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from common.schedules import LinearSchedule
from common.segment_tree import MinSegmentTree, SumSegmentTree

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        """Create Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._memory = []
        self._position = 0
        self.capacity = capacity

    def push(self, *args):
        """Saves a transition."""
        if len(self._memory) < self.capacity:
            self._memory.append(None)
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def _retrieve_sample(self, idxes):
        return [self._memory[idx] for idx in idxes] 

    def __len__(self):
        return len(self._memory)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args, **kwargs):
        idx = self._position
        super().push(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        transitions: [Transition]
            batch of transitions 
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        transitions = self._retrieve_sample(idxes)
        return transitions, (weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

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
        self.writer = SummaryWriter() 
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.dqn_type = config["dqn-type"]
        self.run_title = config["run-title"]
        self.env = gym.make(config["environment"])

        self.num_states  = np.prod(self.env.observation_space.shape)
        self.num_actions = self.env.action_space.n

        layers = [
            self.num_states, 
            *config["architecture"], 
            self.num_actions
        ]

        self.policy_net = Q_Network(layers).to(self.device)
        self.target_net = Q_Network(layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        capacity = config["max-experiences"]
        self.p_replay_eps = config["p-eps"]
        self.prioritized_replay = config["prioritized-replay"]
        self.replay_buffer = PrioritizedReplayBuffer(capacity, config["p-alpha"]) if self.prioritized_replay \
                 else ReplayBuffer(capacity)

        self.beta_scheduler = LinearSchedule(config["episodes"], initial_p=config["p-beta-init"], final_p=1.0)
        self.epsilon_decay = lambda e: max(config["epsilon-min"], e * config["epsilon-decay"])

        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch-size"]
        self.time_step = 0

        self.optim = T.optim.AdamW(self.policy_net.parameters(), lr=config["lr-init"], weight_decay=config["weight-decay"])
        self.lr_scheduler = T.optim.lr_scheduler.StepLR(self.optim, step_size=config["lr-step"], gamma=config["lr-gamma"])
        self.criterion = nn.SmoothL1Loss(reduction="none") # Huber Loss
        self.min_experiences = max(config["min-experiences"], config["batch-size"])

        self.save_path = config["save-path"]

    def act(self, state, epsilon=0):
        """
            Act on environment using epsilon-greedy
        """
        if np.random.sample() < epsilon:
            return int(np.random.choice(np.arange(self.num_actions)))
        else:
            self.policy_net.eval()
            return self.policy_net(T.tensor(state, device=self.device).float()).argmax().item()

    def soft_update(self, tau):
        """
            Polyak averaging: soft update model parameters. 
            θ_target = τ*θ_current + (1 - τ)*θ_target
        """
        for target_param, current_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*target_param.data + (1.0-tau)*current_param.data)

    def optimize(self, beta=None):
        if len(self.replay_buffer) < self.min_experiences:
            return None, None 

        self.policy_net.train()

        if self.prioritized_replay:
            transitions, (is_weights, t_idxes) = self.replay_buffer.sample(self.batch_size, beta)
        else:
            transitions = self.replay_buffer.sample(self.batch_size)
            is_weights, t_idxes = T.ones(self.batch_size), None

        # transpose the batch --> transition of batch-arrays
        batch = Transition(*zip(*transitions))
        # compute a mask of non-final states and concatenate the batch elements
        non_final_mask = T.tensor(tuple(map(lambda state: state is not None, batch.next_state)), 
                                                                device=self.device, dtype=T.bool)  
        non_final_next_states = T.cat([T.tensor([state]).float() for state in batch.next_state if state is not None]).to(self.device)

        state_batch  = T.tensor(batch.state,  device=self.device).float()
        action_batch = T.tensor(batch.action, device=self.device).long()
        reward_batch = T.tensor(batch.reward, device=self.device).float()

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
        next_state_values = T.zeros(self.batch_size, device=self.device)
        if self.dqn_type == "DDQN":
            self.policy_net.eval()
            action_next_state = self.policy_net(non_final_next_states).max(1)[1]
            self.policy_net.train()
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, action_next_state.unsqueeze(1)).squeeze().detach()
        else:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # compute the expected Q values (RHS of the Bellman equation)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # compute temporal difference error
        td_error = T.abs(state_action_values.squeeze() - expected_state_action_values).detach().cpu().numpy()

        # compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = T.mean(loss * T.tensor(is_weights, device=self.device))
      
        # optimize the model
        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        return td_error, t_idxes

    def run_episode(self, epsilon, beta):
        total_reward, done = 0, False
        state = self.env.reset()
        while not done:
            # use epsilon-greedy to get an action
            action = self.act(state, epsilon)
            # caching the information of current state
            prev_state = state
            # take action
            state, reward, done, _ = self.env.step(action)
            # accumulate reward
            total_reward += reward
            # store the transition in buffer
            if done: state = None 
            self.replay_buffer.push(prev_state, action, state, reward)
            # optimize model
            td_error, t_idxes = self.optimize(beta=beta)
            # update target network
            self.soft_update(self.tau)
            # update priorities 
            if self.prioritized_replay and td_error is not None:
                self.replay_buffer.update_priorities(t_idxes, td_error + self.p_replay_eps)
            # increment time-step
            self.time_step += 1

        return total_reward

    def train(self, episodes, epsilon, solved_reward):
        total_rewards = np.zeros(episodes)
        for episode in range(episodes):
            
            # compute beta using linear scheduler
            beta = self.beta_scheduler.value(episode)
            # run episode and get rewards
            reward = self.run_episode(epsilon, beta)
            # exponentially decay epsilon
            epsilon = self.epsilon_decay(epsilon)
            # reduce learning rate by
            self.lr_scheduler.step()

            total_rewards[episode] = reward
            avg_reward = total_rewards[max(0, episode-100):(episode+1)].mean()
            last_lr = self.lr_scheduler.get_last_lr()[0]

            self.writer.add_scalar(f'{self.run_title}/reward', reward, episode)
            self.writer.add_scalar(f'{self.run_title}/reward_100', avg_reward, episode)
            self.writer.add_scalar(f'{self.run_title}/lr', last_lr, episode)
            self.writer.add_scalar(f'{self.run_title}/epsilon', epsilon, episode)

            print(f"Episode: {episode} | Last 100 Average Reward: {avg_reward:.5f} | Learning Rate: {last_lr:.5E} | Epsilon: {epsilon:.5E}", end='\r')

            if avg_reward > solved_reward:
                break
        
        self.writer.close()
        print(f"Environment solved in {episode} episodes")
        T.save(self.policy_net.state_dict(), os.path.join(self.save_path, f"{self.run_title}.pt"))

    def visualize(self, load_path=None):
        done = False
        state = self.env.reset()

        if load_path is not None:
            self.policy_net.load_state_dict(T.load(load_path, map_location=self.device))
        self.policy_net.eval()
        
        while not done:
            self.env.render()
            action = self.act(state)
            state, _, done, _ = self.env.step(int(action))
            sleep(0.01) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="configs/dqn.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    agent = DQN(config)

    if config["train"]:
        agent.train(episodes=config["episodes"], epsilon=config["epsilon-start"], solved_reward=config["solved-criterion"])
    
    if config["visualize"]:
        for _ in range(config["vis-episodes"]):
            agent.visualize(config["load-path"])