import numpy as np
import cv2

"""
Minimal example for pile simulation. 
"""

from rope_sim import RopeSim

sim = RopeSim() # initialize sim.
count = 0

while(True):
    # compute random actions.
    u = [1.2, 1.2]
    sim.update(u)

    # save screenshot
    # image = sim.get_current_image()
    # print(image.dtype)
    #cv2.imwrite("screenshot.png", sim.get_current_image())
    count = count + 1

    # refresh rollout every 10 timesteps.
    if (count % 300 == 0):
        sim.refresh()
