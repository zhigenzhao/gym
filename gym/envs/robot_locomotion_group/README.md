# Robot Locomotion Group Gym Folder

Here you will find newly added environments that the RLG group is working on.

# Organization

Generally, the folders here are divided by which simulator the environment is built on. This is so that users can keep simulator dependencies to a minimum depending on which environment they want to use. (For example, if someone just wants to run an instance of carrots running on pymunk, they shouldn't have to install drake).

If your environment requires a new simulator (e.g. new environment built on FleX), then feel free to create a new folder.

# How to add new environments.

See OpenAI's instructions on [creating a new environment](https://github.com/openai/gym/blob/master/docs/creating-environments.md). Here are group specific instructions:

1. Create a folder under `envs/robot_locomotion_group/[simulator]` where `[simulator]` is the simulator you are using. The folder's name should be the new environment name, and you can develop your environment there that contains your own `gym.Env` class.
2. Register your new environment class under `envs/robot_locomotion_group/[simulator]/__init__.py`.
3. Register your new environment name under `envs/__init__.py`. You will see a section dedicated to Robot Locomotion Group. 
4. Now you can use your environment from any location with `gym.make("my_awesome_new_environment-v0")`
5. [IMPORTANT] To test your environment, you can deploy a random agent by `python3 examples/agents/random_agent.py ENV_NAME`
6. If you have time, it's always good to add a `README` to your environment! @mntan3 did a great job on the [shoe environment](https://github.com/RobotLocomotion/gym/tree/master/gym/envs/robot_locomotion_group/drake/shoe).
