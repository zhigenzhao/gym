from future.utils import iteritems
import math
import numpy as np
import os

from pydrake.multibody.parsing import Parser

from gym.envs.robot_locomotion_group.drake.shoe.generate_sdf import SDFGenerator


def generate_rope_sdf_from_config(rope_config, rope_name):
    gen = SDFGenerator(model_name=rope_name,
                       num_segments=rope_config["num_segments"],
                       segment_length=rope_config["segment_length"],
                       link_mass=rope_config["link_mass"],
                       rope_radius=rope_config["rope_radius"],
                       joint_limit=rope_config["joint_limit"],
                       joint_damping=rope_config["joint_damping"],
                       joint_friction=rope_config["joint_friction"],
                       spring_stiffness=rope_config["spring_stiffness"],
                       rope_height=rope_config["rope_height"],
                       rope_x=rope_config["rope_x"],
                       rope_y=rope_config["rope_y"])
    return gen.get_rope_sdf_string()

def filter_collisions(rope_config, plant, scene_graph, rope_names):
    for rope_name in rope_names:
        for i in range(1, rope_config["num_segments"]):
            body_a = plant.GetBodyByName(f"{rope_name}_capsule_{i}")
            set_a = plant.CollectRegisteredGeometries([body_a])

            body_b = plant.GetBodyByName(f"{rope_name}_capsule_{i + 1}")
            set_b = plant.CollectRegisteredGeometries([body_b])
            scene_graph.ExcludeCollisionsBetween(set_a, set_b)

def add_ground(plant, scene_graph):
    parser = Parser(plant, scene_graph)
    base_folder = os.path.abspath(os.path.dirname(__file__))
    ground_file = os.path.join(base_folder, "model/capsule.frag.xml")
    ground_model = parser.AddModelFromFile(ground_file)
    return ground_model

def add_rope(plant, scene_graph, rope_config, X_W, rope_name="rope"):
    parser = Parser(plant, scene_graph)
    rope_sdf = generate_rope_sdf_from_config(rope_config, rope_name)
    rope_model = parser.AddModelFromString(file_contents=rope_sdf, file_type="sdf", model_name=rope_name)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(f"{rope_name}_capsule_1"), X_W)
    return rope_model

def get_rope_info(diagram, simulator, plant, model_index):
    simulator_context = simulator.get_mutable_context()
    plant_context = diagram.GetMutableSubsystemContext(plant, simulator_context)
    q = plant.GetPositions(plant_context, model_index)
    print(q)

def initialize_rope_zero(diagram, simulator, station, rope_name):
    simulator_context = simulator.get_mutable_context()
    station_context = diagram.GetMutableSubsystemContext(station, simulator_context)

    rope_model = station.model_ids[rope_name]
    q_len = station.mbp.num_positions(rope_model)
    v_len = station.mbp.num_velocities(rope_model)
    q_rope = np.zeros(q_len)
    q_rope[0] = -math.pi / 2

    station.set_model_state(station_context, rope_name, q_rope, np.zeros(v_len))

def post_finalize_rope_settings(config, mbp, sg):
    mbp.set_penetration_allowance(
        config["env"]["penetration_allowance"]
    )
    mbp.set_stiction_tolerance(
        config["env"]["stiction_tolerance"]
    )

    filter_collisions(
        config["rope"], mbp, sg, list(config['env']['ropes'].keys()))
