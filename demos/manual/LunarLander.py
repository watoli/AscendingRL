import gymnasium as gym
import matplotlib
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from PIL import Image
import os

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 设置 GIF 路径为相对于项目根目录
gif_path = os.path.join(project_root, 'results', 'manual', 'lunar_lander.gif')


# 设置 matplotlib 字体，以正常显示中文字符
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams["axes.unicode_minus"] = False

# 初始化 pygame 用于捕捉键盘输入
pygame.init()

# 设置显示窗口
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Lunar Lander Controller")

# 定义键位映射
action_map = {
    pygame.K_w: 0,  # 无操作
    pygame.K_a: 3,  # 向左旋转引擎
    pygame.K_s: 2,  # 主引擎
    pygame.K_d: 1   # 向右旋转引擎
}

# 捕捉用户输入并映射到对应的动作
def get_user_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        return 'exit'
    for key, action in action_map.items():
        if keys[key]:
            return action
    # 默认无操作
    return 0

# 重置环境并根据指定的初始位置进行设置
def reset_environment(env, initial_position, random_seed=123, difficulty=1):
    np.random.seed(random_seed)
    observation, _ = env.reset(seed=random_seed)
    if initial_position == 'random':
        return observation

    # 简单模式：无风、无湍流、重力标准、零初始速度和角度
    if difficulty == 1:
        env.unwrapped.gravity = -10.0
        env.unwrapped.wind_power = 0.0
        env.unwrapped.turbulence_power = 0.0
        env.unwrapped.lander.position = np.array([10.0, 5.0])
        env.unwrapped.lander.velocity = np.array([0.0, 0.0])
        env.unwrapped.lander.angle = 0.0
        env.unwrapped.lander.angularVelocity = -1.0

    # 中等模式：轻微的风和湍流、较小的初始速度和角度变化
    elif difficulty == 2:
        env.unwrapped.gravity = -10.0
        env.unwrapped.wind_power = 5.0
        env.unwrapped.turbulence_power = 0.5
        env.unwrapped.lander.position = np.random.uniform(low=[-0.5, 0.5], high=[0.5, 1.5])
        env.unwrapped.lander.velocity = np.random.uniform(low=[-1.0, -1.0], high=[1.0, 1.0])
        env.unwrapped.lander.angle = np.random.uniform(low=-0.2, high=0.2)
        env.unwrapped.lander.angularVelocity = np.random.uniform(low=-0.2, high=0.2)

    # 困难模式：强风和湍流、随机较大的初始速度和角度变化
    elif difficulty == 3:
        env.unwrapped.gravity = -12.0
        env.unwrapped.wind_power = 15.0
        env.unwrapped.turbulence_power = 1.5
        env.unwrapped.lander.position = np.random.uniform(low=[-1.5, -1.5], high=[1.5, 1.5])
        env.unwrapped.lander.velocity = np.random.uniform(low=[-5.0, -5.0], high=[5.0, 5.0])
        env.unwrapped.lander.angle = np.random.uniform(low=-np.pi / 4, high=np.pi / 4)
        env.unwrapped.lander.angularVelocity = np.random.uniform(low=-1.0, high=1.0)

    return observation

# 输出游戏结束时的详细信息
def print_landing_info(terminated, truncated, reward, observation):
    print("=====================================")
    print("游戏结束！")

    # 提取观测信息中的关键变量
    x_pos, y_pos, x_vel, y_vel, angle, angular_velocity, leg1_contact, leg2_contact = observation
    angle_degrees = np.degrees(angle)

    # 输出最终的速度、角度、位置和接触信息，并加上单位
    print(f"最终位置: ({x_pos:.2f} m, {y_pos:.2f} m)")
    print(f"水平速度: {x_vel:.2f} m/s, 垂直速度: {abs(y_vel):.2f} m/s {'向下' if y_vel < 0 else '向上'}")
    print(f"最终角度: {angle_degrees:.2f} 度, 角速度: {angular_velocity:.2f} 度/秒")
    print(f"左腿接触: {'是' if leg1_contact == 1 else '否'}, 右腿接触: {'是' if leg2_contact == 1 else '否'}")

    # 根据腿部接触和姿态进行判定
    if terminated:
        if leg1_contact == 1 and leg2_contact == 1:
            if abs(angle_degrees) < 15 and abs(x_vel) < 1.0 and abs(y_vel) < 1.0:
                print("着陆成功！姿态很好，表现优秀！")
            elif abs(angle_degrees) < 30 and abs(x_vel) < 1.5 and abs(y_vel) < 1.5:
                print("着陆成功！姿态和速度略有偏差，但总体表现不错。")
            else:
                print("着陆成功！速度或角度偏差较大，着陆较为惊险。")
        elif leg1_contact == 1 or leg2_contact == 1:
            if abs(angle_degrees) < 20 and abs(x_vel) < 1.5 and abs(y_vel) < 1.5:
                print("部分接触地面，姿态不错，稍加改进可成功。")
            else:
                print("部分接触地面，但姿态或速度存在较大问题，需要改进。")
        else:
            if abs(angle_degrees) < 20 and abs(x_vel) < 1.5 and abs(y_vel) < 1.5:
                print("未接触地面，但姿态较好，差点就成功了。")
            else:
                print("坠毁！姿态和速度均不理想，未能成功着陆。")
    elif truncated:
        print("由于时间限制或飞出边界，游戏提前结束。")
    else:
        print("未能成功着陆。")

    # 输出最终的总奖励
    print(f"总奖励: {reward:.2f} 分")
    print("=====================================")

# 绘制着陆数据
def plot_landing_data(times, heights, speeds, angles, angular_velocities, output_path):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(times, heights, label='高度')
    plt.xlabel('时间 (秒)')
    plt.ylabel('高度 (米)')
    plt.title('高度随时间变化')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(times, speeds, label='速度', color='orange')
    plt.xlabel('时间 (秒)')
    plt.ylabel('速度 (米/秒)')
    plt.title('速度随时间变化')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(times, angles, label='角度', color='green')
    plt.xlabel('时间 (秒)')
    plt.ylabel('角度 (度)')
    plt.title('角度随时间变化')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(times, angular_velocities, label='角速度', color='red')
    plt.xlabel('时间 (秒)')
    plt.ylabel('角速度 (度/秒)')
    plt.title('角速度随时间变化')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'landing_data.png'))
    plt.close()

# 保存操作动画为 GIF
def save_gif(frames, output_path):
    with Image.new('RGB', (600, 400)) as img:
        img.save(output_path, save_all=True, append_images=frames, duration=100, loop=0)

# Mode A: 异步输入
def mode_a(env, initial_position, random_seed, difficulty, save_gif_path, save_plots=True):
    reset_environment(env, initial_position, random_seed, difficulty)
    done = False
    total_reward = 0
    terminated = False
    truncated = False
    observation = None

    # 用于记录数据的列表
    times = []
    heights = []
    speeds = []
    angles = []
    angular_velocities = []

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

        # 以异步方式更新环境，无操作时执行默认动作0
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        # 捕获当前屏幕内容作为 GIF 帧
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(Image.fromarray(frame))

        # 记录数据
        current_time = time.time() - start_time
        x_pos, y_pos, x_vel, y_vel, angle, angular_velocity, _, _ = observation
        times.append(current_time)
        heights.append(y_pos)
        speeds.append(np.sqrt(x_vel ** 2 + y_vel ** 2))
        angles.append(np.degrees(angle))
        angular_velocities.append(np.degrees(angular_velocity))

        done = terminated or truncated
        time.sleep(0.1)  # 添加一个小延时确保控制更加平滑

    print_landing_info(terminated, truncated, total_reward, observation)

    # 生成 GIF
    if save_gif_path:
        save_gif(frames, save_gif_path)

    # 生成数据图表
    if save_plots:
        plot_landing_data(times, heights, speeds, angles, angular_velocities, os.path.dirname(save_gif_path))


# Mode B: 同步输入与更新
def mode_b(env, initial_position, random_seed, difficulty, save_gif_path, save_plots=True):
    reset_environment(env, initial_position, random_seed, difficulty)
    done = False
    total_reward = 0
    terminated = False
    truncated = False
    observation = None

    # 用于记录数据的列表
    times = []
    heights = []
    speeds = []
    angles = []
    angular_velocities = []

    # 用于保存 GIF 帧的列表
    frames = []

    # 时间戳初始化
    start_time = time.time()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 等待用户输入，只有输入时才进行一步更新
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = get_user_action()
            if action == 'exit':
                pygame.quit()
                return

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        # 捕获当前屏幕内容作为 GIF 帧
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(Image.fromarray(frame))

        # 记录数据
        current_time = time.time() - start_time
        x_pos, y_pos, x_vel, y_vel, angle, angular_velocity, _, _ = observation
        times.append(current_time)
        heights.append(y_pos)
        speeds.append(np.sqrt(x_vel ** 2 + y_vel ** 2))
        angles.append(np.degrees(angle))
        angular_velocities.append(np.degrees(angular_velocity))

        done = terminated or truncated
        time.sleep(0.1)  # 可以调整步长来控制更新速度

    print_landing_info(terminated, truncated, total_reward, observation)

    # 生成 GIF
    if save_gif_path:
        save_gif(frames, save_gif_path)

    # 生成数据图表
    if save_plots:
        plot_landing_data(times, heights, speeds, angles, angular_velocities, os.path.dirname(save_gif_path))


# 主函数，用于运行 Mode A 或 Mode B
def run_lunar_lander(mode='A', initial_position='standard', random_seed=42, difficulty=1, save_gif=True,
                     save_plots=True, save_path=gif_path):
    # 初始化 Lunar Lander 环境
    env = gym.make('LunarLander-v2', render_mode='human')

    # 创建输出目录（如果不存在）
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if mode == 'A':
        print("运行于模式A: 异步输入")
        mode_a(env, initial_position, random_seed, difficulty, save_path if save_gif else None, save_plots)
    elif mode == 'B':
        print("运行于模式B: 同步输入")
        mode_b(env, initial_position, random_seed, difficulty, save_path if save_gif else None, save_plots)
    else:
        print("无效的模式选择！")


# 运行游戏
run_lunar_lander(mode='A', initial_position='standard', random_seed=42, difficulty=1, save_gif=True, save_plots=True)
