import numpy as np

def _skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def base_to_end_rotation(E, B=None):
    """Computes the rotation matrix transforming from base to end-effector frame."""
    if B is None:
        B = np.eye(3)
    R = np.dot(B, E.T)
    return R

def distance_vector_from_to(p_target, p_source=None):
    """Computes the translation vector between two points."""
    if p_source is None:
        p_source = np.array([0, 0, 0])
    return np.subtract(p_target, p_source)

def base_to_end_effector_transform(E, p_target, B=None, p_source=None):
    """Computes the transformation matrix transforming from base to end-effector frame."""
    R = base_to_end_rotation(E, B)
    p = distance_vector_from_to(p_target, p_source)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def rotation_matrix_from_omega_theta(omega, theta):
    omega_cross = _skew(omega)
    return np.eye(3) + np.sin(theta) * omega_cross + (1 - np.cos(theta)) * np.dot(omega_cross, omega_cross)

def translation_vector_from_omega_theta_v(omega, theta, v):
    omega_cross = _skew(omega)
    I = np.eye(3)
    p = (I*theta + (1 - np.cos(theta)) * omega_cross + (theta - np.sin(theta))* np.dot(omega_cross, omega_cross))
    return np.matmul(p,v)

def screw_axis_to_transform(screw, theta):
    omega = screw[:3]
    v = screw[3:]
    R = rotation_matrix_from_omega_theta(omega, theta)
    p = translation_vector_from_omega_theta_v(omega, theta, v)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def adjoint_transform(T):
    R = T[:3, :3]
    p = T[:3, 3]
    p_skew = _skew(p)

    top = np.hstack((R, np.zeros((3,3))))
    bottom = np.hstack((p_skew @ R, R))
    return np.vstack((top, bottom))