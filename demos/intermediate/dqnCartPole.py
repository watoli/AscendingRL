import gymnasium as gym

env = gym.make("CartPole-v1")

"""
observation: 推车的位置、推车的速度、杆的角度以及杆角速度
info: 一个空字典
"""
observation, info = env.reset(seed=42)

print(f"""
Environment Information:
-------------------------------------
Observation Space: {env.observation_space} (Shape: {env.observation_space.shape})
Action Space: {env.action_space} (Number of Actions: {env.action_space.n})
Current Observation: {observation}
Additional Info: {info}
"""
)

steps=0

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)