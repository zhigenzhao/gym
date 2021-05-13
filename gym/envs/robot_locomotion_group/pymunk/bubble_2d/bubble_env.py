import numpy as np
import gym 
from gym import error, spaces, utils 
from gym.utils import seeding

from gym.envs.robot_locomotion_group.pymunk.bubble_2d.bubble_sim import BubbleSim

class BubbleEnv(gym.Env):
    metadata = {'render:modes': ['human']}

    def __init__(self):
        self.sim = BubbleSim()

        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0, -0.05]),
            high=np.array([2.0, 2.0, 0.05]), shape=(3,),
            dtype=np.float32
        )
        # No observation space due to variable length.

    def step(self, action):
        for _ in range(10):
            body_position_lst, done = self.sim.update(action)
            obs = (body_position_lst,
                   np.array([self.sim.body.position, self.sim.body.angle]))
        # We'll return the actual image here instead of the normalized one.
        return obs, 0.0, done, {}

    def reset(self):
        self.sim.refresh()
        return None

    def render(self, mode='human'):
        # Return full resolution image for debugging / rendering.
        return None

    def close(self):
        pass
