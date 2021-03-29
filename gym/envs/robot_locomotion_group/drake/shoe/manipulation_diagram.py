from future.utils import iteritems
import os
import math
import numpy as np
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from pydrake.geometry import (
    Box,
    Cylinder,
    DrakeVisualizer,
    GeometryFrame,
    GeometryInstance,
    MakePhongIllustrationProperties,
    SceneGraph,
    Sphere)
from pydrake.geometry.render import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.manipulation.schunk_wsg import SchunkWsgPositionController
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant
)
from pydrake.systems.sensors import CameraInfo
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer, ConnectMeshcatVisualizer
from pydrake.systems.primitives import (Adder, Demultiplexer, MatrixGain,
                                        Multiplexer,
                                        PassThrough,
                                        StateInterpolatorWithDiscreteDerivative)
from pydrake.systems.sensors import RgbdSensor

from gym.envs.robot_locomotion_group.drake.shoe.vis_object import VisObject

from gym.envs.robot_locomotion_group.drake.shoe.rope_utils import (
    add_ground,
    add_rope
)
from gym.envs.robot_locomotion_group.drake.shoe.paths import (
    iiwa_7_no_collision_path,
    iiwa14_no_collision_path,
    iiwa_7_no_collision_path,
    iiwa14_poly_collision_path,
    iiwa14_primitive_collision_path,
    iiwa14_sphere_collision_path,
    iiwa_7_box_collision_path,
    schunk_path,
    franka_arm_path,
    franka_hand_path,
    franka_combined_path
)

from gym.envs.robot_locomotion_group.drake.shoe.drake_utils import get_label_db
from gym.envs.robot_locomotion_group.drake.shoe.camera_utils import camera_transform_from_dict
from gym.envs.robot_locomotion_group.drake.shoe.transform_utils import transform_from_dict

class ManipulationDiagram(Diagram):
    def __init__(self, config):
        Diagram.__init__(self)

        dt = config['env']['mbp_dt']
        builder = DiagramBuilder()
        mbp, sg = AddMultibodyPlantSceneGraph(builder, dt)

        self._config = config
        self._mbp = mbp
        self._sg = sg
        self._builder = builder
        self._rgbd_sensors = dict()
        self._finalized = False

        self._model_ids = dict()
        self._control_mbp = dict()
        self._port_names = []
        self._finalize_functions = []
        self._model_names_to_mask = list()

        # add rendered
        # Add renderer to scene
        renderer_params = RenderEngineVtkParams()
        self._renderer_name = "vtk_renderer"
        self._sg.AddRenderer(
            self._renderer_name, MakeRenderEngineVtk(renderer_params))

    # === Property accessors ========================================
    @property
    def mbp(self):
        return self._mbp

    @property
    def sg(self):
        return self._sg

    @property
    def model_ids(self):
        return self._model_ids

    @property
    def port_names(self):
        return self._port_names

    def get_control_mbp(self, name):
        return self._control_mbp[name]


    # === Add physical components ===================================
    def add_procedurally_generated_table(self):
        mbp = self._mbp
        dims = self._config['env']['table']['size']

        box_shape = Box(*dims)
        T_W_B = RigidTransform(p=np.array([0., 0., -dims[2]/2.]))

        # This rigid body will be added to the world model instance since
        # the model instance is not specified.
        box_body = mbp.AddRigidBody("table", SpatialInertia(
            mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
        mbp.WeldFrames(mbp.world_frame(), box_body.body_frame(), T_W_B)

        color = np.array(self._config['env']['table']['color'])
        mbp.RegisterVisualGeometry(
            box_body, RigidTransform(), box_shape, "table_vis", color)
        friction_params = self._config['env']['table']['coulomb_friction']
        mbp.RegisterCollisionGeometry(
            box_body, RigidTransform(), box_shape, "table_collision",
            CoulombFriction(*friction_params))

    def add_arms_from_config(self, config):
        X_iiwa_wsg = RigidTransform(RollPitchYaw(np.pi/2., 0, np.pi/2.), [0, 0, 0.114])
        X_franka = RigidTransform(RollPitchYaw(0, 0, -np.pi/4.), [0, 0, 0])

        for arm_name, arm_config in iteritems(config['env']['arms']):
            X_arm = transform_from_dict(arm_config)

            if arm_config["arm_type"] == "iiwa_7":
                if arm_config["collision_model"] == "none":
                    iiwa7_path = iiwa_7_no_collision_path
                elif arm_config["collision_model"] == "box":
                    iiwa7_path = iiwa_7_box_collision_path
                self.add_arm_gripper(arm_name, iiwa7_path, "iiwa_link_0", X_arm,
                                     schunk_path, "iiwa_link_7",
                                     "body", X_iiwa_wsg)
            elif arm_config["arm_type"] == "iiwa_14":
                if arm_config["collision_model"] == "none":
                    iiwa14_path = iiwa14_no_collision_path
                elif arm_config["collision_model"] == "sphere":
                    iiwa14_path = iiwa14_sphere_collision_path
                elif arm_config["collision_model"] == "cylinder":
                    iiwa14_path = iiwa14_primitive_collision_path
                elif arm_config["collision_model"] == "ee_mesh":
                    iiwa14_path = iiwa14_poly_collision_path
                self.add_arm_gripper(arm_name, iiwa14_path, "base", X_arm,
                                     schunk_path, "iiwa_link_7",
                                     "body", X_iiwa_wsg)
            elif arm_config["arm_type"] == "franka":
                self.add_arm_gripper(arm_name, franka_arm_path,
                                     "panda_link0", X_arm,
                                     franka_hand_path, "panda_link8",
                                     "panda_hand", X_franka)
            elif arm_config["arm_type"] == "floating":
                self.add_floating_gripper(arm_name, schunk_path, None, "body", X_arm)

    @staticmethod
    def calculate_ee_composite_inertia(gripper_path):
        plant = MultibodyPlant(0)
        parser = Parser(plant)
        parser.AddModelFromFile(gripper_path)
        plant.Finalize()

        if gripper_path == schunk_path:
            gripper_body = plant.GetBodyByName("body")
            left_finger = plant.GetBodyByName("left_finger")
            right_finger = plant.GetBodyByName("right_finger")
            left_joint = plant.GetJointByName("left_finger_sliding_joint")
            right_joint = plant.GetJointByName("right_finger_sliding_joint")
        elif gripper_path == franka_hand_path:
            gripper_body = plant.GetBodyByName("panda_hand")
            left_finger = plant.GetBodyByName("panda_leftfinger")
            right_finger = plant.GetBodyByName("panda_rightfinger")
            left_joint = plant.GetJointByName("panda_finger_joint1")
            right_joint = plant.GetJointByName("panda_finger_joint2")
        else:
            raise ValueError("Gripper %s not known" % gripper_path)

        X_FLP = left_joint.frame_on_parent().GetFixedPoseInBodyFrame()
        X_FLC = left_joint.frame_on_child().GetFixedPoseInBodyFrame()
        X_FL = X_FLP.multiply(X_FLC.inverse())
        X_FRP = right_joint.frame_on_parent().GetFixedPoseInBodyFrame()
        X_FRC = right_joint.frame_on_child().GetFixedPoseInBodyFrame()
        X_FR = X_FRP.multiply(X_FRC.inverse())

        I_base = gripper_body.default_spatial_inertia()
        I_FL = left_finger.default_spatial_inertia()
        I_FR = right_finger.default_spatial_inertia()

        I_ee = I_base
        I_ee += I_FL.ReExpress(X_FL.rotation()).Shift(-X_FL.translation())
        I_ee += I_FR.ReExpress(X_FR.rotation()).Shift(-X_FR.translation())
        return I_ee

    def add_floating_gripper(self, gripper_name, gripper_path, arm_end_frame, gripper_base, X_gripper):
        parser = Parser(self._mbp, self._sg)
        gripper_model_id = parser.AddModelFromFile(gripper_path, gripper_name)
        self._model_ids[gripper_name] = gripper_model_id

        ee_base_frame = self._mbp.GetFrameByName(gripper_base, gripper_model_id)
        if arm_end_frame is not None:
            self._mbp.WeldFrames(arm_end_frame, ee_base_frame, X_gripper)

        # Add gripper controller stack
        gripper_controller = self._builder.AddSystem(
            SchunkWsgPositionController(0.001, 4000, 5))
        mbp_to_gripper_gain = np.array([[-1., 1., 0., 0.], [0., 0., -1., 1.]])
        mbp_to_gripper_state = self._builder.AddSystem(
            MatrixGain(mbp_to_gripper_gain))

        def finalize_func():
            builder = self._builder
            builder.Connect(gripper_controller.get_generalized_force_output_port(),
                            self._mbp.get_actuation_input_port(gripper_model_id))
            # Add Gripper ports
            if arm_end_frame is None:
                # Gripper is not welded
                # State goes [_, r, p, y, x, y, z, g1, g2, dr, dp, dy, dx, dy, dz, dg1, dg2]
                # Gripper state
                demux = self._builder.AddSystem(Demultiplexer([1, 6, 2, 6, 2]))
                mux = self._builder.AddSystem(Multiplexer(input_sizes=[2, 2]))
                builder.Connect(self._mbp.get_state_output_port(gripper_model_id),
                                demux.get_input_port(0))
                builder.Connect(demux.get_output_port(2),
                                mux.get_input_port(0))
                builder.Connect(demux.get_output_port(4),
                                mux.get_input_port(1))
                builder.Connect(mux.get_output_port(0),
                                gripper_controller.get_state_input_port())
                builder.Connect(mux.get_output_port(0),
                                mbp_to_gripper_state.get_input_port(0))
            else:
                builder.Connect(self._mbp.get_state_output_port(gripper_model_id),
                                gripper_controller.get_state_input_port())
                builder.Connect(self._mbp.get_state_output_port(gripper_model_id),
                                mbp_to_gripper_state.get_input_port(0))

            gripper_position_name = gripper_name + "_position"
            gripper_force_limit_name = gripper_name + "_force_limit"
            gripper_state_name = gripper_name + "_state_measured"
            gripper_force_meas_name = gripper_name + "_force_measured"
            gripper_gen_force_meas_name = gripper_name + "_gen_force_measured"
            gripper_full_state = gripper_name + "_full_state"
            self._port_names.extend([gripper_position_name, gripper_force_limit_name,
                                     gripper_state_name, gripper_force_meas_name,
                                     gripper_gen_force_meas_name, gripper_full_state])
            builder.ExportInput(gripper_controller.get_desired_position_input_port(),
                                gripper_position_name)
            builder.ExportInput(gripper_controller.get_force_limit_input_port(),
                                gripper_force_limit_name)
            builder.ExportOutput(mbp_to_gripper_state.get_output_port(0),
                                    gripper_state_name)
            builder.ExportOutput(gripper_controller.get_grip_force_output_port(),
                                    gripper_force_meas_name)
            builder.ExportOutput(gripper_controller.get_generalized_force_output_port(),
                                    gripper_gen_force_meas_name)
            builder.ExportOutput(self._mbp.get_state_output_port(gripper_model_id),
                                 gripper_full_state)
            if arm_end_frame is None:
                gripper_spatial_position = gripper_name + "_spatial_position"
                gripper_spatial_velocity = gripper_name + "_spatial_velocity"
                self._port_names.extend([gripper_spatial_position, gripper_spatial_velocity])
                builder.ExportOutput(demux.get_output_port(1),
                                     gripper_spatial_position)
                builder.ExportOutput(demux.get_output_port(3),
                                     gripper_spatial_velocity)

        self._finalize_functions.append(finalize_func)

    def add_arm_gripper(self, arm_name, arm_path, arm_base, X_arm,
                        gripper_path, arm_ee, gripper_base, X_gripper):
        # Add arm
        parser = Parser(self._mbp, self._sg)
        arm_model_id = parser.AddModelFromFile(arm_path, arm_name)
        arm_base_frame = self._mbp.GetFrameByName(arm_base, arm_model_id)
        self._mbp.WeldFrames(self._mbp.world_frame(), arm_base_frame, X_arm)
        self._model_ids[arm_name] = arm_model_id

        # Add gripper
        gripper_name = arm_name+"_gripper"
        arm_end_frame = self._mbp.GetFrameByName(arm_ee, arm_model_id)
        self.add_floating_gripper(gripper_name, gripper_path, arm_end_frame, gripper_base, X_gripper)
        

        # Add arm controller stack
        ctrl_plant = MultibodyPlant(0)
        parser = Parser(ctrl_plant)
        ctrl_arm_id = parser.AddModelFromFile(arm_path, arm_name)
        arm_base_frame = ctrl_plant.GetFrameByName(arm_base, ctrl_arm_id)
        ctrl_plant.WeldFrames(ctrl_plant.world_frame(), arm_base_frame, X_arm)

        gripper_equivalent = ctrl_plant.AddRigidBody(
            gripper_name+"_equivalent", ctrl_arm_id,
            self.calculate_ee_composite_inertia(gripper_path))
        arm_end_frame = ctrl_plant.GetFrameByName(arm_ee, ctrl_arm_id)
        ctrl_plant.WeldFrames(arm_end_frame, gripper_equivalent.body_frame(),
                              X_gripper)

        ctrl_plant.Finalize()
        self._control_mbp[arm_name] = ctrl_plant
        arm_num_positions = ctrl_plant.num_positions(ctrl_arm_id)
        kp = 4000*np.ones(arm_num_positions)
        ki = 0 * np.ones(arm_num_positions)
        kd = 5*np.sqrt(kp)
        arm_controller = self._builder.AddSystem(InverseDynamicsController(
            ctrl_plant, kp, ki, kd, False))
        adder = self._builder.AddSystem(Adder(2, arm_num_positions))
        state_from_position = self._builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                arm_num_positions, self._mbp.time_step(), True))

        # Add command pass through and state splitter
        arm_command = self._builder.AddSystem(PassThrough(arm_num_positions))
        state_split = self._builder.AddSystem(Demultiplexer(
            2*arm_num_positions, arm_num_positions))

        def finalize_func():
            builder = self._builder

            # Export positions commanded
            command_input_name = arm_name + "_position"
            command_output_name = arm_name + "_position_commanded"
            self._port_names.extend([command_input_name, command_output_name])
            builder.ExportInput(arm_command.get_input_port(0), command_input_name)
            builder.ExportOutput(arm_command.get_output_port(0), command_output_name)

            # Export arm state ports
            builder.Connect(self._mbp.get_state_output_port(arm_model_id),
                            state_split.get_input_port(0))
            arm_q_name = arm_name + "_position_measured"
            arm_v_name = arm_name + "_velocity_estimated"
            arm_state_name = arm_name + "_state_measured"
            self._port_names.extend([arm_q_name, arm_v_name, arm_state_name])
            builder.ExportOutput(state_split.get_output_port(0), arm_q_name)
            builder.ExportOutput(state_split.get_output_port(1), arm_v_name)
            builder.ExportOutput(self._mbp.get_state_output_port(arm_model_id),
                                 arm_state_name)

            # Export controller stack ports
            builder.Connect(self._mbp.get_state_output_port(arm_model_id),
                            arm_controller.get_input_port_estimated_state())
            builder.Connect(arm_controller.get_output_port_control(),
                            adder.get_input_port(0))
            builder.Connect(adder.get_output_port(0),
                            self._mbp.get_actuation_input_port(arm_model_id))
            builder.Connect(state_from_position.get_output_port(0),
                            arm_controller.get_input_port_desired_state())
            builder.Connect(arm_command.get_output_port(0),
                            state_from_position.get_input_port(0))
            torque_input_name = arm_name + "_feedforward_torque"
            torque_output_cmd_name = arm_name + "_torque_commanded"
            torque_output_est_name = arm_name + "_torque_measured"
            self._port_names.extend([torque_input_name, torque_output_cmd_name,
                                     torque_output_est_name])
            builder.ExportInput(adder.get_input_port(1), torque_input_name)
            builder.ExportOutput(adder.get_output_port(0), torque_output_cmd_name)
            builder.ExportOutput(adder.get_output_port(0), torque_output_est_name)

            external_torque_name = arm_name + "_torque_external"
            self._port_names.append(external_torque_name)
            builder.ExportOutput(
                self._mbp.get_generalized_contact_forces_output_port(arm_model_id),
                external_torque_name)

        self._finalize_functions.append(finalize_func)

    def add_object_from_file(self, object_name, object_path):
        parser = Parser(self._mbp, self._sg)
        self._model_ids[object_name] = parser.AddModelFromFile(object_path, object_name)
        # this should be masked
        self._model_names_to_mask.append(object_name)

        def finalize_func():
            state_output_port = self._mbp.get_state_output_port(self._model_ids[object_name])
            port_state_output_name = object_name + '_state_output'
            self._port_names.append(port_state_output_name)
            self._builder.ExportOutput(state_output_port, port_state_output_name)

        self._finalize_functions.append(finalize_func)

    def add_rope_and_ground(self, include_ground=True):
        if include_ground:
            ground_model = add_ground(self.mbp, self.sg)
            self._model_ids["ground"] = ground_model
        for rope_name, rope_config in iteritems(self._config['env']['ropes']):
            X_W = transform_from_dict(rope_config)
            rope_model = add_rope(self.mbp, self.sg, self._config['rope'], X_W, rope_name)
            self._model_names_to_mask.append(rope_name)
            self._model_ids[rope_name] = rope_model
            link_length = np.linalg.norm(np.subtract(self._config["env"]["ropes"]["rope_2"]["pos"], self._config["env"]["ropes"]["rope_1"]["pos"]))
            lx, ly, lz = self._config["env"]["ropes"]["rope_1"]["pos"]
            rx, ry, rz = self._config["env"]["ropes"]["rope_2"]["pos"]
            # TODO: not completely right
            roll = math.pi / 2 + math.atan2(rz - lz, ry - ly)
            pitch = 0
            yaw = -math.atan2(rx - lx, ry - ly)
            link_I = SpatialInertia(mass=1,
                        p_PScm_E=np.array([0, 0, 0]),
                        G_SP_E=UnitInertia(1, 1, 1))
            link_name = "middle_link"
            link_body = self._mbp.AddRigidBody(link_name, rope_model, link_I)
            cylinder_geom = Cylinder(self._config["rope"]["rope_radius"], link_length)
            X = RigidTransform()
            self._mbp.RegisterVisualGeometry(link_body, X, cylinder_geom,
                                "middle_vis1", [0.5, 0.5, 0.5, 0.5])
            self._mbp.RegisterCollisionGeometry(link_body, X, cylinder_geom,
                                "middle_collision",
                                CoulombFriction(0.0, 0.0))
            self._mbp.WeldFrames(self._mbp.world_frame(), self._mbp.GetFrameByName(f"middle_link", rope_model), RigidTransform(RollPitchYaw(roll, pitch, yaw), [(lx + rx) / 2.0, (ly + ry) / 2.0, (lz + rz) / 2.0]))
            
    # === Add Cameras ===============================================
    def add_rgbd_sensor(self, camera_name, sensor_config):
        """
        Adds Rgbd camera to the diagram
        """
        builder = self._builder

        width = sensor_config['width']
        height = sensor_config['height']
        fov_y = sensor_config['fov_y']
        z_near = sensor_config['z_near']
        z_far = sensor_config['z_far']
        renderer_name = self._renderer_name

        # This is in right-down-forward convention
        X_W_camera = camera_transform_from_dict(sensor_config)
        color_camera = ColorRenderCamera(
            RenderCameraCore(
                renderer_name,
                CameraInfo(width, height, fov_y),
                ClippingRange(z_near, z_far),
                RigidTransform()
            ), False)
        depth_camera = DepthRenderCamera(color_camera.core(),
                                             DepthRange(z_near, z_far))



        # add camera system
        camera = builder.AddSystem(RgbdSensor(parent_id=self._sg.world_frame_id(), X_PB=X_W_camera,
                        color_camera=color_camera,
                        depth_camera=depth_camera))
        builder.Connect(self._sg.get_query_output_port(),
                        camera.query_object_input_port())

        self._rgbd_sensors[camera_name] = camera

    def add_rgbd_sensors_from_config(self, config):
        if not config['env']['rgbd_sensors']["enabled"]:
            return
        for camera_name, sensor_config in iteritems(config['env']['rgbd_sensors']['sensor_list']):
            self.add_rgbd_sensor(camera_name, sensor_config)

    # === Add visualizers ===========================================
    def connect_to_meshcat(self):
        meshcat = ConnectMeshcatVisualizer(self._builder, scene_graph=self._sg,
                                           zmq_url="tcp://127.0.0.1:6000",
                                           draw_period=1)
        return meshcat

    def connect_to_drake_visualizer(self):
        DrakeVisualizer.AddToBuilder(builder=self._builder, scene_graph=self._sg)

    def add_vis_object(self, name, color):
        source_id = self._sg.RegisterSource(name)
        frame_id = self._sg.RegisterFrame(source_id, GeometryFrame(f"{name}_frame"))
        geom = GeometryInstance(RigidTransform(), Sphere(0.014), f"{name}_geometry")
        geom.set_illustration_properties(
            MakePhongIllustrationProperties(color)
        )
        goal_id = self._sg.RegisterGeometry(source_id, frame_id, geom)
        goal_vis = self._builder.AddSystem(VisObject(frame_id))
        self._builder.Connect(goal_vis.get_output_port(0),
                        self._sg.get_source_pose_port(source_id))
        return goal_vis

    # === Finalize the completed diagram ============================
    def finalize(self):
        self._mbp.Finalize()
        self._finalized = True

        for func in self._finalize_functions:
            func()

        self._port_names.extend(["pose_bundle", "contact_results",
                                 "plant_continuous_state", "geometry_poses", "spatial_input"])
        self._builder.ExportOutput(self._sg.get_pose_bundle_output_port(),
                                   "pose_bundle")
        self._builder.ExportOutput(self._mbp.get_contact_results_output_port(),
                                   "contact_results")
        self._builder.ExportOutput(self._mbp.get_state_output_port(),
                                   "plant_continuous_state")
        self._builder.ExportOutput(self._mbp.get_geometry_poses_output_port(),
                                   "geometry_poses")
        self._builder.ExportInput(self._mbp.get_applied_spatial_force_input_port(),
                                  "spatial_input")
        self._builder.ExportOutput(self._mbp.get_body_poses_output_port(),
                                   "body_poses")

        self._builder.BuildInto(self)

    def is_finalized(self):
        return self._finalized

    # === State getters & setters ===================================
    def get_model_state(self, context, model_name):
        assert self.is_finalized()
        model_idx = self._model_ids[model_name]
        mbp_context = self.GetMutableSubsystemContext(self._mbp, context)

        d = dict()
        d['position'] = np.copy(self._mbp.GetPositions(mbp_context, model_idx))
        d['velocity'] = np.copy(self._mbp.GetVelocities(mbp_context, model_idx))
        return d

    def set_model_state(self, context, model_name, q, v):
        assert self.is_finalized()
        model_idx = self._model_ids[model_name]
        mbp_context = self.GetMutableSubsystemContext(self._mbp, context)
        self._mbp.SetPositions(mbp_context, model_idx, q)
        self._mbp.SetVelocities(mbp_context, model_idx, v)

    def get_pusher_state(self, context):
        return self.get_model_state(context, "pusher")

    def set_pusher_state(self, context, q, v):
        return self.set_model_state(context, "pusher", q, v)

    def get_image_observations_single_sensor(self, sensor_name, context):
        assert self.is_finalized()
        sensor = self._rgbd_sensors[sensor_name]
        sensor_context = self.GetSubsystemContext(sensor, context)
        rgb = sensor.color_image_output_port().Eval(sensor_context)
        depth_32F = sensor.depth_image_32F_output_port().Eval(sensor_context)
        depth_16U = sensor.depth_image_16U_output_port().Eval(sensor_context)
        label = sensor.label_image_output_port().Eval(sensor_context)

        return {'rgb': np.copy(rgb.data[:, :, :3]),
                'depth_32F': np.copy(depth_32F.data).squeeze(),
                'depth_16U': np.copy(depth_16U.data).squeeze(),
                'label': np.copy(label.data).squeeze()}

    def get_image_observations(self, context):
        d = dict()
        for sensor_name in self._rgbd_sensors:
            d[sensor_name] = self.get_image_observations_single_sensor(sensor_name, context)
        return d

    def get_label_db(self):
        return get_label_db(self._mbp)

    def get_labels_to_mask(self):
        """
        Returns a set of labels that represent objects of interest in the scene
        Usually this is the object being pushed
        :return: TinyDB database
        """
        # write sql query
        label_db = self.get_label_db()

        # return all objects that have matching model_name in the list
        Label = Query()
        labels_to_mask = label_db.search(Label.model_name.one_of(self._model_names_to_mask))

        # create new TinyDB with just those
        mask_db = TinyDB(storage=MemoryStorage)
        mask_db.insert_multiple(labels_to_mask)

        return mask_db
