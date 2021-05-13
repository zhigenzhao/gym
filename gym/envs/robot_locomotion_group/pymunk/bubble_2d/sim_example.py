import numpy as np
import cv2

"""
Minimal example for pile simulation. 
"""

from bubble_sim import BubbleSim

sim = BubbleSim() # initialize sim.
count = 0

while(True):
    # compute random actions.
    if (count < 100):
        u = [1.0, 1.0, -0.01]
    else:
        u = [1.0, 1.0, 0.0]
    sim.update(u)

    # save screenshot
    # image = sim.get_current_image()
    # print(image.dtype)
    #cv2.imwrite("screenshot.png", sim.get_current_image())
    count = count + 1

    # refresh rollout every 10 timesteps.
    if (count % 300 == 0):
        sim.refresh()
