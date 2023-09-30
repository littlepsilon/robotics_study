import numpy as np
from functools import reduce
from .utils import adjoint_transform, screw_axis_to_transform, base_to_end_effector_transform

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

    def compute_joint_transform(self, joint_name, isBase=True, isInverse=False):
        if joint_name not in self.joints:
            raise ValueError(f"No joint named {joint_name} in the robot.")
        
        theta = self.joints[joint_name].theta
        screw_axis = self.compute_screw_axis(joint_name, isBase)
        if isInverse:
            return screw_axis_to_transform(-screw_axis, theta)
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


    def compute_jacobian_base(self, joint_thetas, isBase=True):
        self.update_joint_thetas(joint_thetas)

        screw_axes = [self.compute_screw_axis(joint_name, isBase) for joint_name in joint_thetas]
        transformation_matrices = [self.compute_joint_transform(joint_name, isBase, isInverse=(not isBase)) for joint_name in joint_thetas]
        
        traverse_order = range(len(transformation_matrices))
        if not isBase:
            traverse_order = (traverse_order)[::-1]

        jacobian_columns = []
        cur_T = np.eye(4)
        for idx in traverse_order:
            screw_axis = screw_axes[idx]
            adjoint = adjoint_transform(cur_T)
            jacobian_col = np.dot(adjoint, screw_axis).reshape(-1,1)
            jacobian_columns.append(jacobian_col)

            cur_T = np.dot(cur_T, transformation_matrices[idx])

        if not isBase:
            jacobian_columns = (jacobian_columns)[::-1]
            
        J = np.hstack(jacobian_columns)
            
        return J