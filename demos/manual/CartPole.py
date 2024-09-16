import gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from pylab import mpl

# 设置 matplotlib 字体，以正常显示中文字符
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams["axes.unicode_minus"] = False

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 设置 GIF 和图表路径为相对于项目根目录
gif_path = os.path.join(project_root, 'results', 'manual', 'cart_pole.gif')
plot_path = os.path.join(project_root, 'results', 'manual')

# 初始化 pygame
pygame.init()

# 设置显示窗口
screen_width, screen_height = 600, 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("CartPole Controller")

# 定义键位映射
action_map = {
    pygame.K_LEFT: 0,  # 向左推小车
    pygame.K_RIGHT: 1  # 向右推小车
}

def get_user_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        return 'exit'
    for key, action in action_map.items():
        if keys[key]:
            return action
    return 0

def plot_landing_data(times, positions, velocities, output_path):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(times, positions, label='位置')
    plt.xlabel('时间 (秒)')
    plt.ylabel('位置 (米)')
    plt.title('位置随时间变化')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, velocities, label='速度', color='orange')
    plt.xlabel('时间 (秒)')
    plt.ylabel('速度 (米/秒)')
    plt.title('速度随时间变化')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'cart_pole.png'))
    plt.close()

def save_gif(frames, output_path):
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

def run_cartpole(random_seed=42, save_gif_path=None, save_plot_path=None):
    env = gym.make('CartPole-v1', render_mode='human')
    env.reset(seed=random_seed)

    done = False
    total_reward = 0

    # 用于记录数据的列表
    times = []
    positions = []
    velocities = []

    # 用于保存 GIF 帧的列表
    frames = []

    # 时间戳初始化
    start_time = time.time()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 获取用户输入动作
        action = get_user_action()
        if action == 'exit':
            pygame.quit()
            return

        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Render and capture the screen content
        env.render()
        pygame.display.flip()

        # Capture the screen content
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(Image.fromarray(frame))

        # 记录数据
        current_time = time.time() - start_time
        x_pos, x_vel, theta, theta_dot = observation
        times.append(current_time)
        positions.append(x_pos)
        velocities.append(x_vel)

        done = terminated or truncated
        time.sleep(0.1)

    print(f"总奖励: {total_reward:.2f} 分")

    # 生成 GIF
    if save_gif_path:
        save_gif(frames, save_gif_path)

    # 生成数据图表
    if save_plot_path:
        plot_landing_data(times, positions, velocities, save_plot_path)

    env.close()
    pygame.quit()

# 创建输出目录（如果不存在）
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# 运行 CartPole
run_cartpole(random_seed=42, save_gif_path=gif_path, save_plot_path=plot_path)
