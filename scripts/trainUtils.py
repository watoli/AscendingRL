from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self,action,  state,  next_state, reward , terminated , truncated , info ):
        self.buffer.append((action,  state,  next_state, reward , terminated , truncated , info))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        action,  state,  next_state, reward , terminated , truncated , info = zip(*transitions)
        return action, np.array(state), np.array(next_state), reward , terminated , truncated , info

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'terminated': [],'truncated': [], 'info': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    transition_dict['state'].append(state)
                    transition_dict['action'].append(action)
                    transition_dict['reward'].append(reward)
                    transition_dict['next_state'].append(next_state)
                    transition_dict['terminated'].append(terminated)
                    transition_dict['truncated'].append(truncated)
                    transition_dict['info'].append(info)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward , terminated , truncated , info = env.step(action)
                    replay_buffer.add(next_state, reward , terminated , truncated , info)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns,b_te, b_tr, b_i = replay_buffer.sample(batch_size)
                        transition_dict = {'state': b_s, 'action': b_a, 'reward': b_r, 'next_state': b_ns, 'terminated': b_te, 'truncated': b_tr, 'info': b_i}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lamda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lamda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
