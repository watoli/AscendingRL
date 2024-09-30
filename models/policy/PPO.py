import torch
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import delta
from sympy.physics.units import action

import scripts.trainUtils as trainUtils

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    class PPO:
        def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, lmbda, eps, epochs, gamma, device):
            self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
            self.critic = ValueNet(state_dim, hidden_dim).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.lmbda = lmbda
            self.eps = eps
            self.epochs = epochs
            self.gamma = gamma
            self.device = device

        def take_action(self, state):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            probs=self.actor(state) # 计算动作概率
            action_dist = torch.distributions.Categorical(probs) # 生成分布
            action = action_dist.sample() # 采样动作
            return action.item() # 返回动作

        def update(self, transition_dict):
            state = torch.tensor(transition_dict['state'], dtype=torch.float32).to(self.device)
            action = torch.tensor(transition_dict['action'], dtype=torch.long).to(self.device)
            reward = torch.tensor(transition_dict['reward'], dtype=torch.float32).to(self.device)
            next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float32).to(self.device)
            terminated = torch.tensor(transition_dict['terminated'], dtype=torch.bool).to(self.device)
            truncated = torch.tensor(transition_dict['truncated'], dtype=torch.bool).to(self.device)  # 可选，处理截断状态
            info = transition_dict['info']  # 可选，调试信息

            td_target = reward + self.gamma * self.critic(next_state) * (1 - terminated.float()) # 计算TD目标
            td_error = td_target - self.critic(state)
            advantage = trainUtils.compute_advantage(self.gamma, self.lmbda, td_delta)

