from future.utils import iteritems
import math
import numpy as np
import os

from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.math import (
    RigidTransform,
    RollPitchYaw,
)
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import (
    ExternallyAppliedSpatialForce
)
from pydrake.systems.framework import (
    BasicVector,
    LeafSystem
)


class SetpointController(LeafSystem):
    def __init__(self, arms, limit, timestep=0.005):
        LeafSystem.__init__(self)
        self.limit = limit
        self.timestep = timestep
        self.arms = arms

        self.target_input_ports = {}
        self.setpoint_output_ports = {}
        for arm in self.arms:
            self.target_input_ports[arm] = {}
            self.setpoint_output_ports[arm] = {}
            self.target_input_ports[arm]["position"] = self.DeclareVectorInputPort(
                f"{arm}_target",
                BasicVector(6),
            )
            self.target_input_ports[arm]["width"] = self.DeclareVectorInputPort(
                f"{arm}_width_target",
                BasicVector(1),
            )
        if "left" in self.arms:
            self.setpoint_output_ports["left"]["position"] = self.DeclareVectorOutputPort(
                f"left_setpoint",
                BasicVector(6),
                self.DoCalcLeftOutput
            )
            self.setpoint_output_ports["left"]["width"] = self.DeclareVectorOutputPort(
                f"left_width_setpoint",
                BasicVector(1),
                self.DoCalcLeftWidthOutput
            )
        if "right" in self.arms:
            self.setpoint_output_ports["right"]["position"] = self.DeclareVectorOutputPort(
                f"right_setpoint",
                BasicVector(6),
                self.DoCalcRightOutput
            )
            self.setpoint_output_ports["right"]["width"] = self.DeclareVectorOutputPort(
                f"right_width_setpoint",
                BasicVector(1),
                self.DoCalcRightWidthOutput
            )

        self.current_states = {}
        for arm in self.arms:
            self.current_states[arm] = {}
            self.current_states[arm]["position"] = self.DeclareDiscreteState(6)
            self.current_states[arm]["width"] = self.DeclareDiscreteState(1)

        self.DeclarePeriodicDiscreteUpdate(self.timestep)

    def SetPositions(self, context, arm, position, width):
        context.get_mutable_discrete_state(self.current_states[arm]["position"]).SetFromVector(position)
        context.get_mutable_discrete_state(self.current_states[arm]["width"]).SetFromVector(width)

    def DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state):
        for arm in self.arms:
            # Position
            position_target = self.target_input_ports[arm]["position"].Eval(context)
            position_current = context.get_discrete_state(self.current_states[arm]["position"]).get_value()
            position_setpoint = self.get_rpyxyz_setpoint(position_target, position_current)
            discrete_state.get_mutable_vector(self.current_states[arm]["position"]).\
                SetFromVector(position_setpoint)
            
            # Width
            width_target = self.target_input_ports[arm]["width"].Eval(context)
            width_current = context.get_discrete_state(self.current_states[arm]["width"]).get_value()
            width_setpoint = self.get_width_setpoint(width_target, width_current)
            discrete_state.get_mutable_vector(self.current_states[arm]["width"]).\
                SetFromVector(width_setpoint)
            # print(f"Arm {arm}   Position setpoint {position_setpoint}     Width setpoint {width_setpoint}")
    
    def get_width_setpoint(self, target, current):
        displace_limit = self.limit["width"]
        displaced = np.clip(np.subtract(target, current), np.negative(displace_limit), displace_limit)
        setpoint = np.add(displaced, current)
        return setpoint

    def get_rpyxyz_setpoint(self, target, current):
        displace_limit = self.limit["position"]
        diff_arr = np.subtract(target, current)
        s_vals = []
        for i in range(len(diff_arr)):
            if diff_arr[i] != 0:
                s_vals.append(displace_limit[i] / abs(diff_arr[i]))
        if len(s_vals) > 0:
            s = min(s_vals)
        else:
            s = 0

        displace = s * diff_arr
        setpoint = np.add(displace, current)
        # TODO: Do the angle wrap-arounds properly
        setpoint[:3] = target[:3]
        return setpoint

    def DoCalcLeftOutput(self, context, y_data):
        y_data.SetFromVector(context.get_discrete_state(self.current_states["left"]["position"]).get_value())

    def DoCalcRightOutput(self, context, y_data):
        y_data.SetFromVector(context.get_discrete_state(self.current_states["right"]["position"]).get_value())

    def DoCalcLeftWidthOutput(self, context, y_data):
        y_data.SetFromVector(context.get_discrete_state(self.current_states["left"]["width"]).get_value())

    def DoCalcRightWidthOutput(self, context, y_data):
        y_data.SetFromVector(context.get_discrete_state(self.current_states["right"]["width"]).get_value())


class SpatialHandController(LeafSystem):
    def __init__(self, arm_info, timestep=0.0005):
        LeafSystem.__init__(self)
        self.set_name("low_level_hand_controller")
        self.arm_info = arm_info
        self.previous_error = {}
        self.integral = {}
        self.desired_input_ports = {}
        self.estimated_input_ports = {}
        self.body_positions_input_port = self.DeclareAbstractInputPort(
            "body_positions",
            Value[List[RigidTransform]]()
        )
        for arm in self.arm_info:
            self.desired_input_ports[arm] = self.DeclareVectorInputPort(
                f"{arm}_desired",
                BasicVector(6)
            )
            self.previous_error[arm] = self.DeclareDiscreteState(6)
            self.integral[arm] = self.DeclareDiscreteState(6)

        self.DeclareAbstractOutputPort(
            f"spatial_forces_vector",
            lambda: Value[List[ExternallyAppliedSpatialForce]](),
            self.DoCalcAbstractOutput
        )
        self.kp = [100, 100, 100, 10000, 10000, 10000]
        self.ki = [0, 0, 0, 1, 1, 1]
        self.kd = [10, 10, 10, 10, 10, 10]
        self.timestep = timestep

    def reset(self, context):
        for arm in self.arm_info:
            context.get_mutable_discrete_state(self.previous_error[arm]).SetFromVector(np.zeros(6))
            context.get_mutable_discrete_state(self.integral[arm]).SetFromVector(np.zeros(6))

    def get_external_force(self, body_index, force):
        # print(f"Force {force[:3]}")
        external_force = ExternallyAppliedSpatialForce()
        external_force.body_index = body_index
        external_force.p_BoBq_B = np.zeros(3)
        external_force.F_Bq_W = SpatialForce(
            tau=force[0:3], f=force[3:6])
        return external_force

    def clamp_angles(self, rpy):
        adjusted_rpy = []
        for angle in rpy:
            if angle >= math.pi:
                adjusted_rpy.append(angle - 2 * math.pi)
            elif angle < -math.pi:
                adjusted_rpy.append(angle + 2 * math.pi)
            else:
                adjusted_rpy.append(angle)
        return adjusted_rpy

    def DoCalcAbstractOutput(self, context, y_data):
        """
        Sets external forces
        """
        external_forces = []
        body_positions = self.body_positions_input_port.Eval(context)
        for arm in self.arm_info:
            desired = self.desired_input_ports[arm].Eval(context)
            pose = body_positions[int(self.arm_info[arm])]
            rotation = RollPitchYaw(pose.rotation())
            estimated = [rotation.roll_angle(), rotation.pitch_angle(), rotation.yaw_angle()]
            estimated.extend(pose.translation())
            error = np.subtract(desired, estimated)
            error[0:3] = self.clamp_angles(error[0:3])
            prev_error = context.get_mutable_discrete_state(self.previous_error[arm])
            integral = context.get_mutable_discrete_state(self.integral[arm])
            integral.SetFromVector(integral.get_value() + error * self.timestep)
            derivative = np.subtract(
                error, prev_error.get_value()) / self.timestep
            forces = np.multiply(self.kp, error)
            forces += np.multiply(self.kd, derivative)
            forces += np.multiply(self.ki, integral.get_value())

            clipped_forces = np.clip(forces, [-100, -100, -100, -10000, -10000, -10000], [100, 100, 100, 10000, 10000, 10000])
            prev_error.SetFromVector(error)
            # print(f"arm {arm}    forces {clipped_forces}")
            external_forces.append(self.get_external_force(self.arm_info[arm], clipped_forces))
        y_data.set_value(external_forces)

def set_targets(simulator, diagram, systems, values):
    simulator_context = simulator.get_mutable_context()
    for key in systems["targets"]:
        context = diagram.GetMutableSubsystemContext(systems["targets"][key], simulator_context)
        setpoint = systems["targets"][key].get_mutable_source_value(context)
        setpoint.get_mutable_value()[:] = values[key]

def modify_targets(simulator, diagram, systems, values):
    simulator_context = simulator.get_mutable_context()
    new_values = {}
    for key in systems["targets"]:
        context = diagram.GetMutableSubsystemContext(systems["targets"][key], simulator_context)
        setpoint = systems["targets"][key].get_mutable_source_value(context)
        setpoint.get_mutable_value()[:] = setpoint.get_value() + values[key]
        new_values[key] = setpoint.get_mutable_value()[:]
    return new_values
