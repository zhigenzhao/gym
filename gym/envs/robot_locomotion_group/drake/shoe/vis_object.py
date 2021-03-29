import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.geometry import (FramePoseVector, GeometryFrame, GeometryInstance,
                              MakePhongIllustrationProperties, SceneGraph, Sphere)
from pydrake.math import RigidTransform
from pydrake.systems.framework import LeafSystem

class VisObject(LeafSystem):
    def __init__(self, frame_id):
        LeafSystem.__init__(self)
        self.DeclareAbstractOutputPort(
            "goal", lambda: AbstractValue.Make(FramePoseVector()),
            self.OutputGoal)
        self.goal = RigidTransform(np.zeros(3))
        self.frame_id = frame_id

    def OutputGoal(self, context, poses):
        goal_pose = FramePoseVector()
        goal_pose.set_value(self.frame_id, self.goal)
        poses.set_value(goal_pose)

    def SetGoal(self, goal):
        self.goal = RigidTransform(np.array(goal))
