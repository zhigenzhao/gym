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
from pydrake.systems.analysis import Simulator, ResetIntegratorFromFlags, GetIntegrationSchemes

from .utils import HopperObservationLogger, hopper_running_cost

class HopperDrakeEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config=None):
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            config = yaml.safe_load(open(config_path, 'r'))
        
        self._config = config
        self._step_dt = config["step_dt"]
        self._model_name = config["model_name"]
        if "integrator" in config and config["integrator"] in GetIntegrationSchemes():
            self._integration_scheme = config["integrator"]
        else:
            print("Using default integration scheme.")
            self._integration_scheme = "implicit_euler"

        self._sim_diagram = DrakeSimDiagram(config)
        mbp = self._sim_diagram.mbp
        builder = self._sim_diagram.builder

        # add ground
        green = np.array([.5, 1., .5, 1.])
        mbp.RegisterVisualGeometry(
            mbp.world_body(), RigidTransform(), HalfSpace(), "ground_vis", green
        )
        static_friction = config["friction"]
        ground_friction = CoulombFriction(static_friction, static_friction)
        mbp.RegisterCollisionGeometry(
            mbp.world_body(), RigidTransform(), HalfSpace(), "ground_collision", ground_friction
        )

        # add hopper
        parser = Parser(mbp, self._sim_diagram.sg)
        hopper_urdf = os.path.join(os.path.dirname(__file__), "urdf", "footedhopper.urdf")
        
        self._hopper_id = parser.AddModelFromFile(hopper_urdf)
        hopper_base_frame = mbp.GetBodyByName("xtrans", self._hopper_id).body_frame()
        mbp.WeldFrames(mbp.world_frame(), hopper_base_frame, RigidTransform())

        def hopper_port_func():
            actuation_input_port = mbp.get_actuation_input_port(self._hopper_id)
            state_output_port = mbp.get_state_output_port(self._hopper_id)
            contact_output_port = mbp.get_contact_results_output_port()
            builder.ExportInput(actuation_input_port, "hopper_actuation_input_port")
            builder.ExportOutput(state_output_port, "hopper_state_output_port")
            builder.ExportOutput(contact_output_port, "hopper_contact_output_port")
            
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
        mbp.set_penetration_allowance(config["penetration_allowance"])
        mbp.set_stiction_tolerance(config["stiction_tolerance"])

        builder = DiagramBuilder()
        builder.AddSystem(self._sim_diagram)
        builder.ExportInput(self._sim_diagram.GetInputPort("hopper_actuation_input_port"))
        self._diagram = builder.Build()

        self._hopper_actuation_input_port = self._diagram.get_input_port(0)
        self._simulator = Simulator(self._diagram)

        # self.reset()

        self.action_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float32 
        )

        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(10,), dtype=np.float32
        )

        self.obs = {}
        self.logger = HopperObservationLogger(self)
    
    def reset(self, x0=0.):
        # TODO: reset to a random initial state
        assert self._sim_diagram.is_finalized()

        context = self._simulator.get_mutable_context()
        mbp_context = self._diagram.GetMutableSubsystemContext(
            self._sim_diagram.mbp, context)
        context.SetTime(0.)

        q0_hopper = [x0, 1.5, 0.72273432, -1.44546857, 2.29353058]

        self._sim_diagram.mbp.SetPositions(mbp_context, self._hopper_id, q0_hopper)
        self._sim_diagram.mbp.SetVelocities(mbp_context, self._hopper_id, 
            np.zeros(self._sim_diagram.mbp.num_velocities(self._hopper_id)))
        self._hopper_actuation_input_port.FixValue(context, np.zeros(3))

        self._simulator.Initialize()
        self.obs = {}
        self.logger = HopperObservationLogger(self)

        return self.get_observation()

    def get_observation(self, context=None):
        assert self._sim_diagram.is_finalized()
        if context is None:
            context = self._simulator.get_context()
        context = self._diagram.GetMutableSubsystemContext(self._sim_diagram, context)
        mbp_context = self._sim_diagram.GetMutableSubsystemContext(
            self._sim_diagram.mbp, context)
        
        if "control" not in self.obs:
            self.obs["control"] = None
        for key in self.obs.keys():
            if key != "control":
                self.obs[key] = None

        self.obs["time"] = context.get_time()
        self.obs["position"] = np.copy(self._sim_diagram.mbp.GetPositions(
            mbp_context, self._hopper_id
        ))
        self.obs["velocity"] = np.copy(self._sim_diagram.mbp.GetVelocities(
            mbp_context, self._hopper_id
        ))
        self.obs["qv"] = np.copy(self._sim_diagram.mbp.GetPositionsAndVelocities(
            mbp_context, self._hopper_id
        ))

        # get contact forces
        contact_output_port = self._sim_diagram.mbp.get_contact_results_output_port()
        contact_results = contact_output_port.Eval(mbp_context)
        
        contact_forces = {}
        contact_penetration = {}
        for i in range(contact_results.num_point_pair_contacts()):
            contact_info = contact_results.point_pair_contact_info(i)
            point_pair = contact_info.point_pair()
            key = (point_pair.id_A.get_value(), point_pair.id_B.get_value())

            contact_forces[key] = contact_info.contact_force()
            contact_penetration[key] = point_pair.depth
        
        self.obs["contact_forces"] = contact_forces
        self.obs["contact_penetration"] = contact_penetration

        return self.obs

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
        
        self.obs["control"] = action
        obs = self.get_observation()
        obs["running_cost"] = hopper_running_cost(obs["qv"], obs["control"])
        self.logger.add_observation(obs)

        reward = obs["running_cost"]
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

    @property
    def step_dt(self):
        return self._step_dt