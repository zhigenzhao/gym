import gym
import numpy as np
import os
import yaml

from gym.envs.robot_locomotion_group.shoe.open_loop import (
    get_instructions,
    instructions_to_x
)

shoe_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(shoe_dir, 'config.yaml')
config = yaml.safe_load(open(config_path, 'r'))
env = gym.make("Shoe-v0", config=config)
env.reset()

open_loop = get_instructions()
i = 0
action = None
while True:
    obs, _, success, _ = env.step(action)
    action = None
    # If grippers not moving, do next stage
    if np.linalg.norm(obs) < 0.02:
        if i >= len(open_loop):
            print("Done")
            break
        print(f"Executing move {i}")
        action = instructions_to_x([open_loop[i]])
        i += 1

    if not success:
        break