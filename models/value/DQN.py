import collections
import numpy as np
import torch
import torch.nn.functional

from scripts.trainUtils import ReplayBuffer


class Qnet(torch.nn.Module):
    def __init__(self, state_dim,hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim) # 输出层，为每一个动作输出一个Q值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.device = device
        self.state_dim = state_dim # 状态维度
        self.hidden_dim = hidden_dim # 隐藏层维度
        self.action_dim = action_dim # 动作维度

        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device) # Q网络
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(device) # 目标网络

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate) # 目标网络和Q网络同步
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon贪心策略
        self.target_update = target_update # 目标网络同步频率

        self.count=0 # 计数器, 记录更新次数

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_value = self.q_net(state)
            return torch.argmax(q_value).item() # 返回Q值最大的动作

    def update(self, transition_dict):
        # Extract transition components
        state = torch.tensor(transition_dict['state'], dtype=torch.float32).to(self.device)
        action = torch.tensor(transition_dict['action'], dtype=torch.long).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float32).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float32).to(self.device)
        terminated = torch.tensor(transition_dict['terminated'], dtype=torch.bool).to(self.device)
        truncated = torch.tensor(transition_dict['truncated'], dtype=torch.bool).to(self.device)  # 可选，处理截断状态
        info = transition_dict['info']  # 可选，调试信息

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - terminated.float()) * self.gamma * next_q_value

        # 计算当前Q值
        current_q_values = self.q_net(state)
        current_q_value = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # 计算损失
        loss = torch.nn.functional.mse_loss(current_q_value, target_q_value)

        # 更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.count += 1
        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(collections.OrderedDict(self.q_net.state_dict()))

