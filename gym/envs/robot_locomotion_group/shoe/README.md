# Shoe tying environment

## Installation
Dependencies:
- [Python3](https://www.python.org/downloads/)
- [Drake](https://drake.mit.edu/installation.html) for the core simulation
- [pip](https://pip.pypa.io/en/stable/) packages: yaml, numpy, meshcat, future, tinydb, matplotlib

Instructions:
1. Install dependencies above
2. Do gym setup [here](https://github.com/RobotLocomotion/gym/tree/4d8d08c15596acf7bab540c65564545ea2a2307c)


## Running
### Visualization
Choose one of the following
- Run Drake visualizer(recommended): `./bazel-bin/tools/drake-visualizer` or `/opt/drake/bin/drake-visualizer`
- Run meshcat:
  - Change the `meshcat` flag to True
  - Open browser at `http://127.0.0.1:7000/static/` after running program
### Random actions
Run `python3 examples/agents/random_agent.py Shoe-v0` to see random actions applied to the gripper

### Shoe tie
Run `python3 gym/envs/robot_locomotion_group/shoe/test_gym.py` to see a full shoe tie
![2021-03-29-1617040582_screenshot_736x420](https://user-images.githubusercontent.com/13571695/112881825-28df0580-909a-11eb-8733-e8ba96b4f7f6.jpg)
