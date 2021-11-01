import os
import numpy as np
import yaml

import gym
from gym import spaces
from gym.envs.robot_locomotion_group.drake.drake_sim_diagram import DrakeSimDiagram

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform
from pydrake.geometry import HalfSpace
from pydrake.multibody.plant import CoulombFriction
from pydrake.systems.analysis import Simulator

class HopperDrakeEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config=None):
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = yaml.safe_load(open(config_path, 'r'))
        
        self._config = config
        self._step_dt = config["step_dt"]
        self._model_name = config["model_name"]
        self._sim_diagram = DrakeSimDiagram(config)
        mbp = self._sim_diagram.mbp
        builder = self._sim_diagram.builder

        # add ground
        green = np.array([.5, 1., .5, 1.])
        mbp.RegisterVisualGrometry(
            mbp.world_body(), RigidTransform(), HalfSpace(), "ground_vis", green
        )
        static_friction = 1.0
        ground_friction = CoulombFriction(static_friction, static_friction)
        mbp.RegisterCollisionGeometry(
            mbp.world_body(), RigidTransform(), HalfSpace(), "ground_collision", ground_friction
        )

        # add hopper
        parser = Parser(mbp, self._sim_diagram.sg)
        hopper_urdf = os.path.join(os.path.dirname(__file__), "urdf", "footedhopper.urdf")

        self._hopper_id = parser.AddModelFromFile(
            FindResourceOrThrow(hopper_urdf)
        )

        def hopper_port_func():
            actuation_input_port = mbp.get_actuation_input_port(self._hopper_id)
            state_output_port = mbp.get_state_output_port(self._hopper_id)
            builder.ExportInput(actuation_input_port, "hopper_actuation_port")
            builder.ExportOutput(state_output_port, "hopper_state_port")
        self._sim_diagram.finalize_functions.append(hopper_port_func)

        if "visualization" in config:
            if not config["visualization"]:
                pass
            elif config["visualization"] == "meshcat":
                self._sim_diagram.connect_to_meshcat()
            elif config["visualization"] == "drake_viz":
                self._sim_diagram.connect_to_drake_visualizer()
            else:
                raise ValueError("Unknown visualization:", config["visualization"])

        self._sim_diagram.finalize()

        builder = DiagramBuilder()
        builder.AddSystem(self._sim_diagram)
        builder.ExportInput(self._sim_diagram.GetInputPort("hopper_actuation_port"))
        self._diagram = builder.Build()

        self._hopper_actuation_input_port = self._diagram.get_input_port(0)
        self._simulator = Simulator(self._sim_diagram)

        self.reset()

        self.action_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float32 
        )

        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(10,), dtype=np.float32
        )
    
    def reset(self):
        assert self._sim_diagram.is_finalized()

        context = self._simulator.get_mutable_context()
        mbp_context = self._diagram.GetMutableSubsystemContext(
            self._sim_diagram.mbp, context
        )
        context.SetTime(0.)

        q0_hopper = [0., 1.5, 0.72273432, -1.44546857, 2.29353058]
        self._sim_diagram.mbp.SetPositions(mbp_context, self._hopper_id, q0_hopper)
        self._sim_diagram.mbp.SetVelocities(mbp_context, self._hopper_id, np.zeros(5))
        self._hopper_actuation_input_port.FixValue(context, np.zeros(3))

        self._simulator.Initialize()
        return self.get_observation()

    def get_observation(self, context=None):
        assert self._sim_diagram.is_finalized()
        if context is None:
            context = self._simulator.get_context()
        context = self._diagram.GetMutableSubsystemContext(self._sim_diagram, context)
        mbp_context = self._sim_diagram.GetMutableSubsystemContext(
            self._sim_diagram.mbp, context)
        
        obs = {}
        obs["time"] = context.get_time()
        obs["position"] = np.copy(self._sim_diagram.mbp.GetPositions(
            mbp_context, self._hopper_id
        ))
        obs["velocity"] = np.copy(self._sim_diagram.mbp.GetVelocities(
            mbp_context, self._hopper_id
        ))

        return obs

    def step(self, action):
        assert self._sim_diagram.is_finalized()
        assert len(action) == 3
        dt = self._step_dt

        # the time to advance to
        t_adv = self._simulator.get_context().get_time() + dt
        context = self._simulator.get_mutable_context()

        # set the value of input port
        self._hopper_actuation_input_port.FixValue(context, action)

        # simulate and take observation
        self._simulator.AdvanceTo(t_adv)
        obs = self.get_observation()

        reward = 0.
        done = False
        info = {}

        return obs, reward, done, info


   #TODO: Implement as necessary
    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass