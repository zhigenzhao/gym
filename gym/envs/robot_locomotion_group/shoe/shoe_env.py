from future.utils import iteritems
import meshcat
import numpy as np
import os
import yaml

import gym
from gym import error, spaces, utils
from gym.envs.robot_locomotion_group.shoe.build_shoe_diagram import (
    reset_simulator_from_config,
    build_shoe_diagram
)
from gym.envs.robot_locomotion_group.shoe.floating_hand_controllers import (
    modify_targets
)
from gym.envs.robot_locomotion_group.shoe.state_utils import (
    get_gripper_velocities
)
from gym.envs.robot_locomotion_group.shoe.open_loop import (
    x_to_open_loop_instructions
)
class ShoeEnv(gym.Env):
    def __init__(self, config=None):
        if config is None:
            shoe_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(shoe_dir, 'config.yaml')
            self.config = yaml.safe_load(open(config_path, 'r'))
        else:
            self.config = config
        if self.config["env"]["meshcat"]:
            vis = meshcat.Visualizer()
        self.step_dt = self.config["env"]["step_dt"]
        sim_objects = build_shoe_diagram(self.config)
        self.simulator, self.diagram, self.systems, self.visualizer = sim_objects

        self.action_space = spaces.Box(
            low=-0.1,
            high=0.1, shape=(14,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255, shape=(32,32),
            dtype=np.float32
        )

    def step(self, action):
        """
        Action is the change in setpoint
        """
        if action is not None:
            _, instructions = x_to_open_loop_instructions(action, 1)
            modify_targets(self.simulator, self.diagram, self.systems,
                            instructions[0])
        new_time = self.simulator.get_context().get_time() + self.step_dt
        success = self.simulator.AdvanceTo(new_time)
        new_obs = get_gripper_velocities(self.diagram, self.simulator, self.systems)
        return new_obs, 0, success, {}

    def reset(self, config=None):
        """
        Can optionally reset from a different config to change the
        positions of the ropes or grippers
        """
        if config is None:
            config = self.config
        reset_simulator_from_config(self.config, self.simulator, self.diagram, self.systems)

    def close(self):
        pass
