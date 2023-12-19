import numpy as np
import pybullet as p
import quaternion
from scipy.spatial.transform import Rotation as R

# quaternion format converter => (x, y, z, w) to (w, x, y, z)
def xyzw2wxyz(quat : list or np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

# quaternion format converter => (w, x, y, z) to (x, y, z, w)
def wxyz2xyzw(quat : list or np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])

# pose format converter => (x, y, z, rx, ry, rz) to (x, y, z, quaternion in (x, y, z, w) format)
def pose_6d_to_7d(pose : list or tuple or np.ndarray):
    assert len(pose) == 6, f'input array size should be 6d but got {len(pose)}d'

    pos = pose[:3]
    rot_vec = pose[3:]
    rot_quat = R.from_rotvec(rot_vec).as_quat() # x, y, z, w

    return list(pos) + list(rot_quat)

# pose format converter => (x, y, z, quaternion in (x, y, z, w) format) to (x, y, z, rx, ry, rz)
def pose_7d_to_6d(pose : list or tuple or np.ndarray):
    assert len(pose) == 7, f'input array size should be 7d but got {len(pose)}d'

    pos = pose[:3]
    rot_quat = pose[3:]
    rot_vec = R.from_quat(rot_quat).as_rotvec()

    return list(pos) + list(rot_vec)

# convert 6d pose or 7d pose to 4x4 transformation matrix
def get_matrix_from_pose(pose : list or tuple or np.ndarray) -> np.ndarray:
    assert len(pose) == 6 or len(pose) == 7, f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_rotvec(pose[3:]).as_matrix()
    elif len(pose) == 7:
        rot_m = R.from_quat(pose[3:]).as_matrix()
            
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m

# convert 4x4 transformation matrix to 6d pose or 7d pose 
def get_pose_from_matrix(matrix : list or tuple or np.ndarray, 
                        pose_size : int = 7) -> np.ndarray:

    mat = np.array(matrix)
    assert pose_size == 6 or pose_size == 7, f'pose_size should be 6 or 7, but got {pose_size}'
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'
    
    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()  # x, y, z, w
            
    pose = list(pos) + list(rot)

    return np.array(pose)

# get dense waypoints from start_config to end_config by interpolation
def get_dense_waypoints(start_config : list or tuple or np.ndarray, 
                        end_config : list or tuple or np.ndarray, 
                        resolution : float=0.005):

    assert len(start_config) == 7 and len(end_config) == 7

    d12 = np.asarray(end_config) - np.asarray(start_config)
    d12_pos = d12[:3]
    steps = int(np.ceil(np.linalg.norm(np.divide(d12, resolution), ord=2)))
    obj_init_quat = quaternion.as_quat_array(xyzw2wxyz(start_config[3:]))
    obj_tgt_quat = quaternion.as_quat_array(xyzw2wxyz(end_config[3:]))

    ret = []
    for step in range(steps):
        ratio = (step + 1) / steps
        pos = ratio * d12_pos + np.asarray(start_config[:3])
        quat = quaternion.slerp_evaluate(obj_init_quat, obj_tgt_quat, ratio)
        quat = wxyz2xyzw(quaternion.as_float_array(quat))
        position7d = tuple(pos) + tuple(quat)
        ret.append(position7d)

    return ret

# draw the pose in 7d (x, y, z, quaternion (x, y, z, w) format) or 4x4d (transformation matrix)
def draw_coordinate(pose : np.ndarray or tuple or list, 
                    size : float = 0.1, 
                    color : np.ndarray=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    
    assert (type(pose) == np.ndarray and pose.shape == (4, 4)) or (len(pose) == 7)

    if len(pose) == 7:
        pose = get_matrix_from_pose(pose)

    origin = pose[:3, 3]
    x = origin + pose[:3, 0] * size
    y = origin + pose[:3, 1] * size
    z = origin + pose[:3, 2] * size
    p.addUserDebugLine(origin, x, color[0])
    p.addUserDebugLine(origin, y, color[1])
    p.addUserDebugLine(origin, z, color[2])

def draw_bbox(start : list or tuple or np.ndarray,
              end : list or tuple or np.ndarray):
    
    assert len(start) == 3 and len(end) == 3, f'infeasible size of position, len(position) must be 3'

    points_bb = [
        [start[0], start[1], start[2]],
        [end[0], start[1], start[2]],
        [end[0], end[1], start[2]],
        [start[0], end[1], start[2]],
        [start[0], start[1], end[2]],
        [end[0], start[1], end[2]],
        [end[0], end[1], end[2]],
        [start[0], end[1], end[2]],
    ]

    for i in range(4):
        p.addUserDebugLine(points_bb[i], points_bb[(i + 1) % 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i + 4], points_bb[(i + 1) % 4 + 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i], points_bb[i + 4], [1, 0, 0])

def get_robot_joint_info(robot_id):
    num_joints = p.getNumJoints(robot_id)

    joint_states = p.getJointStates(robot_id, range(0, num_joints))
    joint_poses = [x[0] for x in joint_states]
    joint_names = []
    joint_types = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1]
        joint_type = joint_info[2]
        joint_names.append(joint_name)
        joint_types.append(joint_type)
    
    return joint_names, joint_poses, joint_types