import numpy as np
import gym 
from gym import error, spaces, utils 
from gym.utils import seeding

from gym.envs.robot_locomotion_group.pymunk.rope_2d.rope_sim import RopeSim

class RopeEnv(gym.Env):
    metadata = {'render:modes': ['human']}

    def __init__(self):
        self.sim = RopeSim()

        self.action_space = spaces.Box(
            low=-2.0,
            high=2.0, shape=(2,),
            dtype=np.float32
        )

        # Observation space ommitted due to length variability.

    def step(self, action):
        # Zero-order hold?
        for _ in range(10):
            done = self.sim.update(action)
            obs = (self.sim.collision_pairs, np.array(self.sim.body.position))

        # NOTE: the sim will detect the initial touch on the rope. On the next touch,
        # done will be set to True.
        return obs, 0.0, done, {}

    def reset(self):
        self.sim.refresh()
        return (self.sim.collision_pairs, self.sim.body.position)

    def render(self, mode='human'):
        # TODO(terry-suh): Return full resolution image for debugging / rendering.
        return None 

    def close(self):
        pass
