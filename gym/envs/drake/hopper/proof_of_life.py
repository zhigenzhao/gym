import numpy as np
import gym
import time

env = gym.make('Hopper-v0')
env.reset()

zero_torque = np.zeros(3)
N = 200

input("start?")
for i in range(N):
    obs, reward, done, info = env.step(zero_torque)
    time.sleep(env.step_dt)
    if i%100==0:
        print(obs["contact_forces"])
        print(obs["contact_penetration"])
        print(obs["running_cost"])
env.logger.plot_contact()
env.logger.plot_cum_cost()