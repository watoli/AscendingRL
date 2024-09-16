import gymnasium as gym
import keyboard
import threading
from time import sleep


class LunarLanderControl(threading.Thread):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.daemon = True
        self.running = True
        self.action = None

    def run(self):
        while self.running:
            if keyboard.is_pressed('up'):
                self.action = 0  # do nothing
            elif keyboard.is_pressed('down'):
                self.action = 2  # fire main engine
            elif keyboard.is_pressed('right'):
                self.action = 3  # fire right orientation engine
            elif keyboard.is_pressed('left'):
                self.action = 1  # fire left orientation engine
            elif keyboard.is_pressed('esc'):
                self.running = False
            sleep(0.05)  # 减少CPU占用率


def update_env(env, controller):
    try:
        terminated = False
        truncated = False
        while True:
            if controller.action is not None:
                # 执行用户输入的动作
                observation, reward, terminated, truncated, info = env.step(controller.action)
                controller.action = None  # 清空已执行的动作
            else:
                # 没有用户输入时不执行任何动作
                pass

            # 检查是否到达终止或截断状态
            if terminated or truncated:
                observation, info = env.reset()
                terminated = False
                truncated = False

            # 渲染环境以显示更新后的状态
            env.render()
            sleep(0.05)  # 减少CPU占用率

    except KeyboardInterrupt:
        print("退出游戏")


if __name__ == "__main__":
    env = gym.make('LunarLander-v2', render_mode="human")
    observation, info = env.reset()

    controller = LunarLanderControl(env)
    controller.start()

    update_env(env, controller)

    controller.running = True
    controller.join()
    env.close()