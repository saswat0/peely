# Report

### Learning Algorithm

Converging to an expected optimal policy using traditional Reinforcement Learning methods like Monte-Carlo Methods or Temporal-Difference Methods (involving Q-learning or Sarsamax) becomes a full blown optimization problem if an environment's observation or action-space is continuous. 

In this project, the agent is trained with [Deep Q-Networks (DQN)](https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb), a value-based technique which uses Neural Networks to approximate the action-value function. An experience-replay buffer is introduced to store all the past experiences. After an adequate no. of experiences in replay buffer, batches of samples from it are taken randomly to train the Q-network. There are 2 action-value networks, target_Q and local_Q network. While training, the target_Q network is fixed temporarily and local_Q network is updated for a certain no. of steps; then target_Q network is updated. This is done keeping in mind the non-stationary or unstable target problem.

### Model Architecture (HyperParameters)

> Q-Network

The deep-NN architecture defined in `model.py` consists of 2 hidden layers with [64, 64] nodes & with ReLU activation. The model  operates on fully-connected layers.

> Agent

The agent is defined in `agent.py`.

  - `BUFFER_SIZE` = 1e5 # replay buffer size
  - `BATCH_SIZE` = 64   # minibatch size
  - `GAMMA` = 0.99      # reward discount factor
  - `TAU` = 1e-3        # for soft-update of target parameters
  - `LR` = 5e-4         # learning rate
  - `UPDATE_EVERY` = 4  # how often to update the network
  - `seed` = 42         # for all random choices

> Training

  - `n_episodes` = 1800 # max number of episodes
  - `max_t` = 1000      # max step per episode
  - `eps_start` = 1.0   # epsilon for greedy-epsilon policy
  - `eps_end` = 0.01    # min epsilon
  - `eps_decay` = 0.995 # epsilon decay-rate


### Plot of Rewards

The plot contains information about rewards collected & averaged over last 100 episodes. The environment is considered solved when the average reward is atleast +13 over the last 100 episodes (epochs). The agent solved the environment in 377 episodes with an average reward of 13.01 but has been trained for 1800 episodes to assess if it's actually learning. 

### Scope of Future Work
  > Various optimizations to the original DQN algorithm such as [double DQN](https://towardsdatascience.com/double-deep-q-networks-905dd8325412), [dueling DQN](https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751) & Prioritized Experience Replay can   be implemented.
  
  > The neural network can be deeper & certain hyperparameters can be tuned.
   
  > We can build & use our own environment instead of using the pre-defind environment provided by Unity.
