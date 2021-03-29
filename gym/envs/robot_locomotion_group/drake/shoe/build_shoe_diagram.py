from future.utils import iteritems
import math
import numpy as np
import os

from pydrake.common.cpp_param import List
from pydrake.common.eigen_geometry import Quaternion
from pydrake.common.value import Value
from pydrake.math import (
    RigidTransform,
    RollPitchYaw,
    RotationMatrix
)
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    ExternallyAppliedSpatialForce
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    BasicVector,
    DiagramBuilder,
    LeafSystem
)
from pydrake.systems.primitives import (
    ConstantVectorSource
)

from gym.envs.robot_locomotion_group.drake.shoe.floating_hand_controllers import (
    SpatialHandController,
    SetpointController,
    set_targets
)
from gym.envs.robot_locomotion_group.drake.shoe.rope_utils import (
    post_finalize_rope_settings,
    initialize_rope_zero
)
from gym.envs.robot_locomotion_group.drake.shoe.manipulation_diagram import ManipulationDiagram


def build_shoe_diagram(config):
    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationDiagram(config))
    station.add_rope_and_ground(include_ground=False)
    if 'arms' in config['env']:
        station.add_arms_from_config(config)
    parser = Parser(station.mbp, station.sg)
    shoe_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(shoe_dir, "model/shoe.sdf")
    shoe_model = parser.AddModelFromFile(model_file, "shoe")
    if config["env"]["visualization"]:
        station.connect_to_drake_visualizer()
    visualizer = None
    if "meshcat" in config["env"] and config["env"]["meshcat"]:
        visualizer = station.connect_to_meshcat()
    if config["env"]["parameterization"] == "closed":
        left_rope_point = station.add_vis_object("left_rope", [1, 0, 0, 1])
        right_rope_point = station.add_vis_object("right_rope", [0, 1, 0, 1])
        left_target_point = station.add_vis_object("left_target_point", [1, 0, 0, 1])
        right_target_point = station.add_vis_object("right_target_point", [0, 1, 0, 1])
    if config["env"]["rgbd_sensors"]["enabled"]:
        station.add_rgbd_sensors_from_config(config)

    station.finalize()

    post_finalize_rope_settings(config, station.mbp, station.sg)

    targets = {}

    if 'arms' in config['env']:
        gripper_info = {}
        for arm_name, arm_config in iteritems(config['env']['arms']):
            # Add PID Control
            gripper = station.mbp.GetBodyByName("body", station.model_ids[arm_name])
            gripper_info[arm_name] = gripper.index()

            # Initialize targets from file
            init = config["env"]["arms"][arm_name]["rpy"][:]
            init.extend(config["env"]["arms"][arm_name]["pos"])
            targets[arm_name] = builder.AddSystem(ConstantVectorSource(init))
            width_init = config["env"]["arms"][arm_name]["grip"]
            targets[f"{arm_name}_width"] = builder.AddSystem(ConstantVectorSource([width_init]))
        pid = builder.AddSystem(SpatialHandController(gripper_info))
        builder.Connect(station.GetOutputPort(f"body_poses"), pid.GetInputPort("body_positions"))
        sp_control = builder.AddSystem(SetpointController(gripper_info, {"position": [0.005, 0.005, 0.005, 0.0003, 0.0003, 0.0003],
                                        "width": 0.001}))
        for arm_name, arm_config in iteritems(config['env']['arms']):
            builder.Connect(targets[arm_name].get_output_port(0),
                            sp_control.GetInputPort(f"{arm_name}_target"))
            builder.Connect(sp_control.GetOutputPort(f"{arm_name}_setpoint"),
                            pid.GetInputPort(f"{arm_name}_desired"))
            builder.Connect(targets[f"{arm_name}_width"].get_output_port(0),
                            sp_control.GetInputPort(f"{arm_name}_width_target"))
            builder.Connect(sp_control.GetOutputPort(f"{arm_name}_width_setpoint"),
                            station.GetInputPort(f"{arm_name}_position"))
        builder.Connect(pid.GetOutputPort("spatial_forces_vector"),
                        station.GetInputPort("spatial_input"))
    diagram = builder.Build()

    simulator = Simulator(diagram)
    sim_context = simulator.get_mutable_context()
    station_context = diagram.GetMutableSubsystemContext(station, sim_context)

    systems = {"station": station,
               "targets": targets,
               "sp_control": sp_control,
               "pid": pid}
    if config["env"]["parameterization"] == "closed":
        systems["left_rope"] = left_rope_point
        systems["right_rope"] = right_rope_point
        systems["left_target_point"] = left_target_point
        systems["right_target_point"] = right_target_point
    if 'arms' in config['env']:
        values = {}
        for arm_name, arm_config in iteritems(config['env']['arms']):
            station.GetInputPort(f"{arm_name}_force_limit").FixValue(
                                station_context, 40.)
    simulator.set_target_realtime_rate(config['env']['target_realtime_rate'])
    reset_simulator_from_config(config, simulator, diagram, systems)

    return simulator, diagram, systems, visualizer

def reset_simulator_from_config(config, simulator, diagram, systems):
    for rope_name, _ in iteritems(config['env']['ropes']):
        initialize_rope_zero(diagram, simulator, systems["station"], rope_name)
    sim_context = simulator.get_mutable_context()
    station_context = diagram.GetMutableSubsystemContext(systems["station"], sim_context)
    if 'arms' in config['env']:
        values = {}
        for arm_name, arm_config in iteritems(config['env']['arms']):
            rpy = RollPitchYaw(config["env"]["arms"][arm_name]["rpy"])
            xyz = config["env"]["arms"][arm_name]["pos"]
            init_state = np.append(rpy.vector(), xyz)
            grip = config["env"]["arms"][arm_name]["grip"]
            quat = rpy.ToQuaternion()
            systems["station"].set_model_state(station_context, arm_name,
                                    np.array([quat.w(), quat.x(), quat.y(), quat.z(), xyz[0], xyz[1], xyz[2], -grip/2, grip/2]), np.zeros(8))
            values[arm_name] = init_state
            values[f"{arm_name}_width"] = grip
    set_targets(simulator, diagram, systems, values)
    sp_context = diagram.GetMutableSubsystemContext(systems["sp_control"], sim_context)
    if 'arms' in config['env']:
        for arm_name, arm_config in iteritems(config['env']['arms']):
            systems["sp_control"].SetPositions(sp_context, arm_name,
                np.append(config["env"]["arms"][arm_name]["rpy"], config["env"]["arms"][arm_name]["pos"]),
                [config["env"]["arms"][arm_name]["grip"]])

    pid_context = diagram.GetMutableSubsystemContext(systems["pid"], sim_context)
    systems["pid"].reset(pid_context)
    simulator.set_target_realtime_rate(config['env']['target_realtime_rate'])
    sim_context.SetTime(0.)
    simulator.Initialize()