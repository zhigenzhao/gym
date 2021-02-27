import numpy as np
import cv2

"""
Minimal example for pile simulation. 
"""

from gym.envs.robot_locomotion_group.single_integrator.single_integrator_sim import SingleIntegratorSim

sim = SingleIntegratorSim() # initialize sim.
count = 0

while(True):
    # compute random actions.
    u = -1.0 + 2.0 * np.random.rand(2)
    sim.update(u)

    # save screenshot
    image = sim.get_current_image()
    #cv2.imwrite("screenshot.png", sim.get_current_image())
    count = count + 1

    # refresh rollout every 10 timesteps.
    if (count % 200 == 0):
        sim.refresh()
