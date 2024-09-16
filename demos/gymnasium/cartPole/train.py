import torch
from models.value.DQN import DQN
from scripts.trainUtils import ReplayBuffer
import os
import random
import gymnasium as gym

def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dqn = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon_start, target_update, device)
    replay_buffer = ReplayBuffer(capacity=10000)  # 设置回放池容量

    epsilon = epsilon_start
    steps = 0
    best_reward = -float('inf')  # 初始化为负无穷

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            # 动作选择，使用epsilon贪心策略
            if random.random() > epsilon:
                action = dqn.take_action(state)
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # 保存转移数据到Replay Buffer
            replay_buffer.add(action, state, next_state, reward, terminated, truncated, info)

            state = next_state

            # 如果缓冲区足够大则进行训练
            if replay_buffer.size() > batch_size:
                actions, states, next_states, rewards, terminated, truncated, info = replay_buffer.sample(batch_size)
                transition_dict = {
                    'state': states,
                    'action': actions,
                    'reward': rewards,
                    'next_state': next_states,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info
                }
                dqn.update(transition_dict)

            # epsilon 衰减
            steps += 1
            epsilon = max(epsilon_end, epsilon_start - steps / epsilon_decay)

            if terminated or truncated:
                break

        print(f'Episode {episode + 1}, Total Reward: {episode_reward}')

        # 每 10 回合检查是否保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(save_dir, f'cartpole_dqn_best_reward{int(best_reward)}.pth')
            torch.save(dqn.q_net.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path} with reward: {best_reward}')

        # 每 50 回合保存checkpoint
        if (episode + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode + 1}.pth')
            torch.save({
                'episode': episode + 1,
                'model_state_dict': dqn.q_net.state_dict(),
                'target_model_state_dict': dqn.target_net.state_dict(),
                'optimizer_state_dict': dqn.optimizer.state_dict(),
                'epsilon': epsilon
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')


if __name__ == '__main__':
    # 超参数
    learning_rate = 1e-3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 500  # 多少步后 epsilon 衰减到 epsilon_end
    target_update = 100  # 多少次更新后同步目标网络
    buffer_size = 10000  # Replay buffer 大小
    batch_size = 64  # 每次采样的 batch 大小
    max_episodes = 500  # 最大训练回合数
    max_steps_per_episode = 200  # 每回合最大步数
    save_dir = './weights/'  # 模型权重保存位置
    checkpoint_dir = os.path.join(save_dir, 'checkpoints/')  # Checkpoint 保存路径
    os.makedirs(checkpoint_dir, exist_ok=True)

    train()
