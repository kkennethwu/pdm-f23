
import numpy as np
import pybullet as p
import quaternion

from scipy.spatial.transform import Rotation as R

from itertools import product
from pybullet_planning.interfaces.robots.collision import pairwise_link_collision
from pybullet_planning.interfaces.robots.body import set_pose
from pybullet_planning.interfaces.robots.link import get_all_links
from pybullet_planning.interfaces.debug_utils.debug_utils import draw_collision_diagnosis

def get_sample7d_fn(target_conf : list or tuple or np.ndarray,
                    low_limit : list or tuple or np.ndarray,
                    high_limit : list or tuple or np.ndarray, 
                    ratio_to_target=0.1):

    assert len(low_limit) == 3 and len(high_limit) == 3, 'illegal size of limit, len(limit) must be 3'
    
    def sample7d_fn():
        rand_val = np.random.random()
        ret = None
        if rand_val < ratio_to_target:
            ret = target_conf
        else:
            pos_euler = []

            pos_euler.append(np.random.uniform(low_limit[0], high_limit[0])) # x
            pos_euler.append(np.random.uniform(low_limit[1], high_limit[1])) # y 
            pos_euler.append(np.random.uniform(low_limit[2], high_limit[2])) # z
            for i in range(3, 6):
                pos_euler.append(np.random.uniform(-np.pi, np.pi))
            ret = pos_euler[:3] + list(R.from_rotvec(pos_euler[3:]).as_quat())
            
        return tuple(ret)
    return sample7d_fn

def get_distance7d_fn():

    def distance7d_fn(q1, q2):
        assert len(q1) == 7 and len(q2) == 7

        q1_pos = np.asarray(q1[:3])
        q2_pos = np.asarray(q2[:3])
        q1_rot = R.from_quat(np.asarray(q1[3:])).as_rotvec()
        q2_rot = R.from_quat(np.asarray(q2[3:])).as_rotvec()

        diff_pos = q1_pos - q2_pos
        diff_rot = np.array([min((r1 - r2) ** 2, (r2 - r1) ** 2) for r1, r2 in zip(q1_rot, q2_rot)])

        return 1.0 * np.sum(diff_pos ** 2) + 2.0 * np.sum(diff_rot ** 2)
    return distance7d_fn

def xyzw2wxyz(quat : np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray):
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])
    
def get_extend7d_fn(resolution = 0.001):
    
    def extend7d_fn(q1, q2):
        assert len(q1) == 7 and len(q2) == 7

        d12 = np.asarray(q2[:3]) - np.asarray(q1[:3])
        r12 = np.asarray(q2[3:]) - np.asarray(q1[3:])
        diff_q1_q2 = np.concatenate((d12, r12))
        steps = int(np.ceil(np.linalg.norm(np.divide(diff_q1_q2, resolution), ord=2)))
        obj_init_quat = quaternion.as_quat_array(xyzw2wxyz(q1[3:]))
        obj_tgt_quat = quaternion.as_quat_array(xyzw2wxyz(q2[3:]))

        # generate collision check sequence
        yield q1
        for i in range(steps):
            ratio = (i + 1) / steps
            pos = ratio * d12 + np.asarray(q1[:3])
            quat = quaternion.slerp_evaluate(obj_init_quat, obj_tgt_quat, ratio)
            quat = wxyz2xyzw(quaternion.as_float_array(quat))
            positions7d = tuple(pos) + tuple(quat)

            yield positions7d
    
    return extend7d_fn


def expand_links(body):
    """expand all links of a body

    TODO: [REFACTOR] move to body or link modules?

    Parameters
    ----------
    body : int
        [description]

    Returns
    -------
    body : int
        [description]
    links : list of int
        [description]
    """
    body, links = body if isinstance(body, tuple) else (body, None)
    if links is None:
        links = get_all_links(body)
    return body, links

def pairwise_link_collision_info(physicsClientId, body1, link1, body2, link2=-1.0, max_distance=0.0, **kwargs): # 10000
    return p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                              linkIndexA=link1, linkIndexB=link2, physicsClientId=physicsClientId)


def set_pose(physicsClientId, body, pose):
    point = pose[:3]
    quat = pose[3:]
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=physicsClientId)

# TODO: simplified version
def get_collision7d_fn(physicsClientId, body, obstacles=[], attachments=[], disabled_collisions={}, **kwargs):
    """get collision checking function collision_fn(joint_values) -> bool for a floating body (no movable joint).

    Parameters
    ----------
    body : int
        the main moving body (usually the robot). We refer to this body as 'the main body'
        in this function's docstring.
    obstacles : list of int
        body indices for collision objects, by default []
    attachments : list of Attachment, optional
        list of attachment, by default []
    disabled_collisions : set of tuples, optional
        list of tuples for specifying disabled collisions, the tuple must be of the following format:
            ((int, int), (int, int)) : (body index, link index), (body index, link index)
        If the body considered is a single-link (floating) body, assign the link index to BASE_LINK.
        reversing the order of the tuples above is also acceptable, by default {}

    Returns
    -------
    function handle
        collision_fn: (conf, diagnosis) -> False if no collision found, True otherwise.
        if need diagnosis information for the collision, set diagnosis to True will help you visualize
        which link is colliding to which.
    """

    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [body] + attached_bodies
    # * body pairs
    check_body_pairs = list(product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))
    check_body_link_pairs = []
    for body1, body2 in check_body_pairs:
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        bb_link_pairs = product(links1, links2)
        for bb_links in bb_link_pairs:
            bbll_pair = ((body1, bb_links[0]), (body2, bb_links[1]))
            if bbll_pair not in disabled_collisions and bbll_pair[::-1] not in disabled_collisions:
                check_body_link_pairs.append(bbll_pair)

    def collision7d_fn(pose, diagnosis=False):
        set_pose(physicsClientId, body, pose)
        # * body - body check
        for (body1, link1), (body2, link2) in check_body_link_pairs:
            if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
                if diagnosis:
                    # warnings.warn('moving body - body collision!', UserWarning)
                    cr = pairwise_link_collision_info(physicsClientId, body1, link1, body2, link2)
                    draw_collision_diagnosis(cr)
                return True
        return False
    return collision7d_fn


            

        
