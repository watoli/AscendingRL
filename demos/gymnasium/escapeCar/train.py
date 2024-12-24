'''
假设我有这样的任务：
二维平面上物体A与物体B在左右两侧对向行驶，物体A有意撞向物体B（通过追踪法使得速度矢量指向物体B），且物体B可以实时感知到物体A的行驶轨迹。
现需要通过强化学习，训练物体B，使得物体B在尽量在【最晚时间】改变行驶状态躲避物体A（物体B通过状态转移决定行驶状态，如S1表示直行，S2表示左转3度，S3表示右转3度，S4表示加速1m/s，S5表示减速1m/s等等）
物体会有基本的物理约束：1.物体AB有相同的碰撞体积 2. 物体A速度恒定不变 3. 物体A通过追踪法调整方向时每步不超过3度。 4. 物体A、B初始速度均为10m/s，速度最高20m/s，最低为0  5. 初始A、B间隔50米 6. 物体A在横轴驶过物体B（即物体A整个碰撞体积都在物体B右侧）且未发生碰撞时视作物体B规避成功
如何基于gymnasium搭建环境并选择强化学习算法？

请给出搭建环境和强化学习训练代码，
训练好之后，使用权重文件进行决策，并进行可视化展示（即AB两个物体相向行驶，物体B躲避A的过程），
请合并到一个文件里，通过参数自定义训练或演示（或训练完之后自动展示）。
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import matplotlib.backends.backend_agg as agg

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        self.first_turn_step = None
        self.steps = None
        self.done = None
        self.carB_angle = None
        self.carA_angle = None
        self.carB_speed = None
        self.carA_speed = None
        self.carB_pos = None
        self.carA_pos = None
        self.action_space: spaces.Discrete = spaces.Discrete(5)  # S1, S2, S3, S4, S5
        self.observation_space: spaces.Box = spaces.Box(low=np.array([0, 0, -np.pi, 0, 0]), high=np.array([100, 100, np.pi, 20, 20]), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.carA_pos = np.array([0, 50])
        self.carB_pos = np.array([100, 50])
        self.carA_speed = 10
        self.carB_speed = 10
        self.carA_angle = 0
        self.carB_angle = np.deg2rad(180)
        self.done = False
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.carA_pos[0], self.carA_pos[1], self.carA_angle, self.carB_pos[0], self.carB_pos[1]])

    def _update_carA(self):
        direction = self.carB_pos - self.carA_pos
        target_angle = np.arctan2(direction[1], direction[0])
        angle_diff = target_angle - self.carA_angle

        # Normalize the angle difference to the range [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Proportional guidance factor
        k = 0.5  # Adjust this factor as needed for proportional control

        # Calculate the proportional angle change
        proportional_angle_change = k * angle_diff

        # Limit the angle change to a maximum of 2 degrees per step
        max_angle_change = np.deg2rad(2)
        if abs(proportional_angle_change) > max_angle_change:
            proportional_angle_change = np.sign(proportional_angle_change) * max_angle_change

        self.carA_angle += proportional_angle_change
        self.carA_pos[0] += self.carA_speed * np.cos(self.carA_angle)
        self.carA_pos[1] += self.carA_speed * np.sin(self.carA_angle)

    def _update_carB(self, action):
        if action == 0:  # S1: 直行
            self.carB_pos[0] += self.carB_speed * np.cos(self.carB_angle)
            self.carB_pos[1] += self.carB_speed * np.sin(self.carB_angle)
        elif action in [1, 2]:  # S2: 左转, S3: 右转
            if self.first_turn_step is None:
                self.first_turn_step = self.steps  # Record the step of the first turn
            if action == 1:
                self.carB_angle += np.deg2rad(3)
            elif action == 2:
                self.carB_angle -= np.deg2rad(3)
        elif action == 3:  # S4: 加速1m/s
            self.carB_speed = min(self.carB_speed + 1, 20)
        elif action == 4:  # S5: 减速1m/s
            self.carB_speed = max(self.carB_speed - 1, 0)

    def step(self, action):
        initial_distance = np.linalg.norm(self.carA_pos - self.carB_pos)
        initial_x_position = self.carB_pos[0]

        self._update_carB(action)
        self._update_carA()

        self.steps += 1  # 增加步数

        final_distance = np.linalg.norm(self.carA_pos - self.carB_pos)
        final_x_position = self.carB_pos[0]

        if np.linalg.norm(self.carA_pos - self.carB_pos) < 5:
            print("Collision!")
            self.done = True
            reward = -500  # Heavily penalize collisions
        elif self.carA_pos[0] > self.carB_pos[0]:
            self.done = True
            reward = 500  # Fixed reward for successful avoidance
        else:
            reward = 0

            if action in [1, 2]:  # Penalize early turns
                reward -= 5

            if final_distance < initial_distance:  # Penalize moving closer to Car A
                reward += 2

            if final_x_position < initial_x_position:
                reward += 2  # Reward for moving further to the left

            # Additional reward for delaying the first turn
            if self.first_turn_step is not None:
                reward += self.first_turn_step * 2  # Reward based on the step of the first turn

        return self._get_obs(), reward, self.done, {}


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

def save_gif(frames, output_path):
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

def visualize(env, policy_net, save_gif_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net.to(device)
    state = env.reset()
    done = False
    positions_A = [state[:2]]
    positions_B = [state[3:5]]
    frames = []

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy_net(state_tensor).argmax().item()
        print(f"Action: {action}")
        state, _, done, _ = env.step(action)
        positions_A.append(state[:2])
        positions_B.append(state[3:5])

        # Capture the current state of the environment
        fig, ax = plt.subplots()
        ax.plot([pos[0] for pos in positions_A], [pos[1] for pos in positions_A], label='Car A', color='red')
        ax.plot([pos[0] for pos in positions_B], [pos[1] for pos in positions_B], label='Car B', color='blue')
        ax.scatter([0], [50], color='red', label='Start A')
        ax.scatter([100], [50], color='blue', label='Start B')
        ax.legend()
        ax.set_xlim(-10, 110)
        ax.set_ylim(0, 100)
        fig.canvas.draw()

        # Convert the plot to an image
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))
        frames.append(Image.fromarray(image))
        plt.close(fig)

    if save_gif_path:
        save_gif(frames, save_gif_path)

    # Display the final plot
    plt.plot([pos[0] for pos in positions_A], [pos[1] for pos in positions_A], label='Car A')
    plt.plot([pos[0] for pos in positions_B], [pos[1] for pos in positions_B], label='Car B')
    plt.scatter([0], [50], color='red', label='Start A')
    plt.scatter([100], [50], color='blue', label='Start B')
    plt.legend()
    plt.show()
def train_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update, buffer_size, learning_rate):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)

    epsilon = epsilon_start
    steps = 0
    best_reward = -float('inf')

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(200):
            if random.random() > epsilon:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()
            else:
                action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if replay_buffer.size() > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

            steps += 1
            epsilon = max(epsilon_end, epsilon_start - steps / epsilon_decay)

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), model_path)
            print(f"New best model saved with reward: {best_reward}")

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f'Episode {episode + 1}, Total Reward: {episode_reward}')

    return policy_net

if __name__ == '__main__':
    env = CarEnv()
    num_episodes = 500
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1000
    target_update = 10
    buffer_size = 10000
    learning_rate = 1e-3
    model_path = 'car_dqn_model.pth'
    skip_training = False
    gif_path = 'car_simulation.gif'

    if not skip_training:
        policy_net = train_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update, buffer_size, learning_rate)
    else:
        policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
        policy_net.load_state_dict(torch.load(model_path))

    visualize(env, policy_net, save_gif_path=gif_path)