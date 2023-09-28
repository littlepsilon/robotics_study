import numpy as np
from functools import reduce
from .utils import screw_axis_to_transform, base_to_end_effector_transform

class Frame:
    def __init__(self, position=np.zeros(3), orientation=np.eye(3)):
        self.position = np.array(position)
        self.orientation = np.array(orientation)

    def update_position(self, position):
        self.position = np.array(position)

    def update_orientation(self, orientation):
        self.orientation = np.array(orientation)

class Robot:
    def __init__(self):
        self.joints = {}
        self.base_frame = Frame()
        self.end_effector_frame = Frame()

    def register_joint(self, joint):
        self.joints[joint.name] = joint

    def update_joint_thetas(self, joint_thetas):
        for joint_name, theta in joint_thetas.items():
            if joint_name not in self.joints:
                raise ValueError(f"No joint named {joint_name} in the robot.")
            self.joints[joint_name].update_theta(theta)

    def compute_screw_axis(self, joint_name, isBase=True):
        if joint_name not in self.joints:
            raise ValueError(f"No joint named {joint_name} in the robot.")
        reference_frame = self.base_frame if isBase else self.end_effector_frame
        return self.joints[joint_name].compute_screw_axis(reference_frame)

    def compute_joint_transform(self, joint_name, isBase=True):
        if joint_name not in self.joints:
            raise ValueError(f"No joint named {joint_name} in the robot.")
        
        theta = self.joints[joint_name].theta
        screw_axis = self.compute_screw_axis(joint_name, isBase)
        return screw_axis_to_transform(screw_axis, theta)

    def compute_M(self):
        return base_to_end_effector_transform(E=self.end_effector_frame.orientation,
                                              p_target=self.end_effector_frame.position,
                                              B=self.base_frame.orientation,
                                              p_source=self.base_frame.position)

    def compute_total_transform(self, joint_thetas, isBase=True):
        self.update_joint_thetas(joint_thetas)
        
        transformation_matrices = [
            self.compute_joint_transform(joint_name, isBase)
            for joint_name in joint_thetas
        ]
        
        transformation_matrices.append(self.compute_M()) if isBase else transformation_matrices.insert(0, self.compute_M())

        return reduce(np.matmul, transformation_matrices)
