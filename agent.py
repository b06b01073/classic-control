from torch import nn
import torch
import random
import collections
import torch.nn.functional as F
import numpy as np

class Agent():
    def __init__(self, action_dim, obs_dim, eps=1, gamma=0.99, decay=0.995, eps_low = 0.05, batch_size=128, tau=0.75, train_mode=True):
        self.local_network = DQN(action_dim, obs_dim)
        self.target_network = DQN(action_dim, obs_dim)
        self.target_network.load_state_dict(self.local_network.state_dict())
        self.target_network.eval()
        self.losses = []

        self.tau = tau

        self.eps = eps if train_mode else 0

        self.gamma = gamma
        self.decay = decay
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.batch_size = batch_size
        self.eps_low = eps_low


        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.local_network.parameters(), lr=1e-3)


    def step(self, obs):
        with torch.no_grad():
            action_scores = self.local_network.forward(torch.from_numpy(obs))
            return random.choice(range(self.action_dim)) if np.random.uniform() < self.eps else torch.argmax(action_scores).item()


    def eps_decay(self):
        self.eps *= self.decay
        self.eps = max(self.eps, self.eps_low)

    def train(self, replay_buffer):
        self.local_network.train()
        mini_batch = replay_buffer.sample_batch(self.batch_size)

        if mini_batch is None:
            return

        obs, actions, rewards, new_obs, terminals = mini_batch
        Q_local = self.local_network(obs).gather(1, actions.reshape(-1, 1))

        Q_target = torch.max(self.target_network(new_obs), dim=1)[0]
        y = rewards + self.gamma * Q_target * (1 - terminals)

        loss = self.loss_fn(Q_local.reshape(-1), y.reshape(-1))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def soft_update(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.local_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau ) * target_param.data)
     
    def hard_update(self):
        self.target_network.load_state_dict(self.local_network.state_dict())



class DQN(nn.Module):
    def __init__(self, action_dim, obs_dim):
        super().__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)

    def insert(self, item):
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()
        self.buffer.append(item)

    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            return None


        mini_batch = random.sample(self.buffer, batch_size)
        obs = torch.Tensor([experience[0] for experience in mini_batch])
        actions = torch.Tensor([experience[1] for experience in mini_batch]).long()
        rewards = torch.Tensor([experience[2] for experience in mini_batch])
        next_obs = torch.Tensor([experience[3] for experience in mini_batch])
        terminals = torch.Tensor([experience[4] for experience in mini_batch]).long()

        return obs, actions, rewards, next_obs, terminals

    def __len__(self):
        return len(self.buffer)
