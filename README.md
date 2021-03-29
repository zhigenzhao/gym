# OpenAI Gym 

This is a fork of [OpenAI gym](https://github.com/openai/gym) that Robot Locomotion Group uses for custom environments. These environments will include:
- Gym environments wrapped around [Drake](https://drake.mit.edu/)
- Modifications to existing [gym](https://github.com/openai/gym) environments where observations are directly pixels.
- Custom environments written in pymunk.

Some of these environments will be exported to a pip package once they are mature.

# List of New Environments

- Carrot pushing environment. `gym.make("Carrot-v0")`
- Pendulum environment with pixel output. `gym.make("PendulumPixel-v0")`
- Planar single integrator with pixel output. `gym.make("SingleIntegrator-v0")`
- Shoe. `gym.make("Shoe-v0")` or `gym.make("Shoe-v0", config)`

# Setup 

This fork can be setup in the following way.

```
git clone git@github.com:RobotLocomotion/gym.git
cd gym
pip install -e .
``` 

If you are working on this repo then add the following lines:
```
cd gym 
git remote set-url origin git@github.com:RobotLocomotion/gym.git
git remote add upstream git@github.com:openai/gym.git
git remote set-url --push upstream no_push
```

# Workflow 

You can create your own environment on a local branch or your own fork, then PR to this repo once the environment is good enough.
Make sure you **dont't** PR to `openai/gym`, but to `RobotLocomotion/gym`! 

# How to Add a New Environment

To isolate our environments from OpenAI's original environments, let's keep the group environments in `gym/envs/robot_locomotion_group`. You will find detailed instructions in this [folder](https://github.com/RobotLocomotion/gym/tree/master/gym/envs/robot_locomotion_group).

# Dependencies 
- `pyglet`
- `pymunk`
- `drake`
