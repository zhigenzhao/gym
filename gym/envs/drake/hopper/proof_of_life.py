import numpy as np
import gym
import time

env = gym.make('Hopper-v0')
env.reset()

zero_torque = np.zeros(3)
N = 200

input("start?")
for _ in range(N):
    env.step(zero_torque)
    time.sleep(0.03)