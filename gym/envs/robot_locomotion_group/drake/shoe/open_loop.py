import os
import numpy as np


def parameters_to_info(vec):
    # TODO: Do proper scaling for the gripper
    # TODO: Expand scale to -0.3 to 0.3
    info = {"left": {}, "right": {}}
    info["left"]["rpyxyz"] = np.zeros(6)
    info["right"]["rpyxyz"] = np.zeros(6)
    info["left"]["width"] = vec[3] / 3.0
    info["right"]["width"] = vec[7] / 3.0
    info["left"]["rpyxyz"][3:] = vec[:3]
    info["right"]["rpyxyz"][3:] = vec[4: 7]

    return info

def x_to_open_loop_instructions(x, num_moves):
    parameters_per_move = 8
    instructions = []
    for i in range(num_moves):
        info = parameters_to_info(x[parameters_per_move * i: parameters_per_move * i + parameters_per_move])

        instructions.append(
        {"left":  info["left"]["rpyxyz"],
         "right":  info["right"]["rpyxyz"],
         "left_width": info["left"]["width"], "right_width": info["right"]["width"]}
        )
    return x, instructions

def instructions_to_x(instructions):
    x = []
    for instruction in instructions:
        x.append(instruction["left"][3])
        x.append(instruction["left"][4])
        x.append(instruction["left"][5])
        x.append(instruction["left_width"] * 3)
        x.append(instruction["right"][3])
        x.append(instruction["right"][4])
        x.append(instruction["right"][5])
        x.append(instruction["right_width"] * 3)
    return x

def get_instructions():
    # Format of vector is [droll, dpitch, dyaw, dx, dy, dz, dw]
    instructions = [
        {"left":  [0, 0, 0, 0.05, 0, 0],   # Tilt up and align gripper
         "right":  [0, 0, 0, -0.05, 0, 0],
         "left_width": 0, "right_width": 0},
        {"left":  [0, 0, 0, 0, 0.045, -0.04],                    # Go towards shoe
         "right":  [0, 0, 0, 0, -0.045, -0.04],
         "left_width": 0, "right_width": 0},
        {"left":  [0, 0, 0, -0.17, 0, 0],   # Push into rope and starting lifting
         "right":  [0, 0, 0, 0.17, 0, 0],
         "left_width": 0, "right_width": 0},
        {"left":  [0, 0, 0, 0, 0, 0.2],   # Push into rope and starting lifting
         "right":  [0, 0, 0, 0, 0, 0.2],
         "left_width": 0, "right_width": 0},
        {"left":  [0, 0, 0, 0.2, 0.1, 0.0],   # Lift rope, align, grippers come together
         "right":  [0, 0, 0, -0.1, -0, 0.1],
         "left_width": 0.1, "right_width": 0},
        {"left":  [0, 0, 0, 0, 0.1, 0.0],   # Lift rope, align, grippers come together
         "right":  [0, 0, 0, 0, -0.1, 0],
         "left_width": 0, "right_width": 0},
        {"left":  [0, 0, 0, 0, 0, 0],   # Close gripper
         "right":  [0, 0, 0, 0, 0, 0],
         "left_width": -0.1, "right_width": 0},
        {"left":  [0, 0, 0, 0, -0.1, 0.05],   # Align for 2nd rope part 1
         "right":  [0, 0, 0, -0.05, 0., -0.05],
         "left_width": 0, "right_width": 0},
        {"left":  [0, 0, 0, 0, 0, 0.05],   # Align for 2nd rope rope 2
         "right":  [0, 0, 0, 0.0, -0.15, -0.1],
         "left_width": 0, "right_width": 0.1},
        {"left":  [0, 0, 0, 0, 0, 0],   # Close gripper
         "right":  [0, 0, 0, 0, 0, 0],
         "left_width": 0, "right_width": -0.1},
        {"left":  [0, 0, 0, 0, -0.1, -0.05],   # Final rope pull
         "right":  [0, 0, 0, 0, 0.1, 0.1],
         "left_width": 0, "right_width": 0},
        {"left":  [0, 0, 0, -0.1, -0.2, -0.1],   # Final rope pull
         "right":  [0, 0, 0, 0, 0.3, -0.05],
         "left_width": 0, "right_width": 0},
    ]
    return instructions
