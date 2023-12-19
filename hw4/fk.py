import os, argparse, json
import numpy as np

from scipy.spatial.transform import Rotation as R

# for simulator
import pybullet as p

# for geometry information
from hw4_utils.bullet_utils import draw_coordinate, get_matrix_from_pose, get_pose_from_matrix, pose_7d_to_6d, pose_6d_to_7d

SIM_TIMESTEP = 1.0 / 240.0
JACOBIAN_SCORE_MAX = 10.0
JACOBIAN_ERROR_THRESH = 0.05
FK_SCORE_MAX = 10.0
FK_ERROR_THRESH = 0.005
TASK1_SCORE_MAX = JACOBIAN_SCORE_MAX + FK_SCORE_MAX

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    return np.cross(a, b)

def get_ur5_DH_params():

    # TODO: this is the DH parameters (following classic DH convention) of the robot in this assignment,
    # It will be a little bit different from the official spec 
    # You need to use these parameters to compute the forward kinematics and Jacobian matrix
    # details : 
    # see "pybullet_robot_envs/envs/ur5_envs/robot_data/ur5/ur5.urdf" in this project folder
    # official spec : https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    
    dh_params = [
        {'a':  0,      'd': 0.0892,  'alpha':  np.pi/2,  },    # joint1
        {'a':  -0.425, 'd': 0,       'alpha':  0         },    # joint2
        {'a':  -0.392, 'd': 0,       'alpha':  0         },    # joint3
        {'a':  0.,     'd': 0.1093,  'alpha':  np.pi/2   },    # joint4
        {'a':  0.,     'd': 0.09475, 'alpha': -np.pi/2   },    # joint5
        {'a':  0,      'd': 0.2023,  'alpha': 0          },    # joint6
    ]

    return dh_params

def your_fk(DH_params : dict, q : list or tuple or np.ndarray, base_pos) -> np.ndarray:

    # robot initial position
    base_pose = list(base_pos) + [0, 0, 0]  

    assert len(DH_params) == 6 and len(q) == 6, f'Both DH_params and q should contain 6 values,\n' \
                                                f'but get len(DH_params) = {DH_params}, len(q) = {len(q)}'

    A = get_matrix_from_pose(base_pose) # a 4x4 matrix, type should be np.ndarray
    jacobian = np.zeros((6, 6)) # a 6x6 matrix, type should be np.ndarray

    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 1 : Compute Forward-Kinematic and Jacobain of the robot by yourself --- #
    # ---          Try to implement `your_fk` function without using any pybullet  --- #
    # ---          API. (20% for accuracy)                                         --- #
    # -------------------------------------------------------------------------------- #
    
    #### your code ####
    

    # A = ? # may be more than one line
    # jacobian = ? # may be more than one line

    raise NotImplementedError
    # hint : 
    # https://automaticaddison.com/the-ultimate-guide-to-jacobian-matrices-for-robotics/
    
    ###############################################

    # adjustment don't touch
    adjustment = np.asarray([[ 0, -1,  0],
                             [ 0,  0,  0],
                             [ 0,  0, -1]])
    A[:3, :3] = A[:3,:3]@adjustment
    pose_7d = np.asarray(get_pose_from_matrix(A,7))

    return pose_7d, jacobian

# TODO: [for your information]
# This function is the scoring function, we will use the same code 
# to score your algorithm using all the testcases
def score_fk(robot, testcase_files : str, visualize : bool=False):

    testcase_file_num = len(testcase_files)
    dh_params = get_ur5_DH_params()
    fk_score = [FK_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    fk_error_cnt = [0 for _ in range(testcase_file_num)]
    jacobian_score = [JACOBIAN_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    jacobian_error_cnt = [0 for _ in range(testcase_file_num)]

    p.addUserDebugText(text = "Scoring Your Forward Kinematic Algorithm ...", 
                        textPosition = [0.1, -0.6, 1.5],
                        textColorRGB = [1,1,1],
                        textSize = 1.0,
                        lifeTime = 0)

    print("============================ Task 1 : Forward Kinematic ============================\n")
    for file_id, testcase_file in enumerate(testcase_files):

        f_in = open(testcase_file, 'r')
        fk_dict = json.load(f_in)
        f_in.close()
        
        test_case_name = os.path.split(testcase_file)[-1]

        joint_poses = fk_dict['joint_poses']
        poses = fk_dict['poses']
        jacobians = fk_dict['jacobian']

        cases_num = len(fk_dict['joint_poses'])

        penalty = (TASK1_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)

        for i in range(cases_num):
            your_pose, your_jacobian = your_fk(dh_params, joint_poses[i], robot._base_position)
            gt_pose = poses[i]

            if visualize :
                color_yours = [[1,0,0], [1,0,0], [1,0,0]]
                color_gt = [[0,1,0], [0,1,0], [0,1,0]]
                draw_coordinate(your_pose, size=0.01, color=color_yours)
                draw_coordinate(gt_pose, size=0.01, color=color_gt)

            fk_error = np.linalg.norm(your_pose - np.asarray(gt_pose), ord=2)
            
            
     
            if fk_error > FK_ERROR_THRESH:
                fk_score[file_id] -= penalty
                fk_error_cnt[file_id] += 1

            jacobian_error = np.linalg.norm(your_jacobian - np.asarray(jacobians[i]), ord=2)
            if jacobian_error > JACOBIAN_ERROR_THRESH:
                jacobian_score[file_id] -= penalty
                jacobian_error_cnt[file_id] += 1
        
        fk_score[file_id] = 0.0 if fk_score[file_id] < 0.0 else fk_score[file_id]
        jacobian_score[file_id] = 0.0 if jacobian_score[file_id] < 0.0 else jacobian_score[file_id]

        score_msg = "- Testcase file : {}\n".format(test_case_name) + \
                    "- Your Score Of Forward Kinematic : {:00.03f} / {:00.03f}, Error Count : {:4d} / {:4d}\n".format(
                            fk_score[file_id], FK_SCORE_MAX / testcase_file_num, fk_error_cnt[file_id], cases_num) + \
                    "- Your Score Of Jacobian Matrix   : {:00.03f} / {:00.03f}, Error Count : {:4d} / {:4d}\n".format(
                            jacobian_score[file_id], JACOBIAN_SCORE_MAX / testcase_file_num, jacobian_error_cnt[file_id], cases_num)
        
        print(score_msg)
    p.removeAllUserDebugItems()

    total_fk_score = 0.0
    total_jacobian_score = 0.0
    for file_id in range(testcase_file_num):
        total_fk_score += fk_score[file_id]
        total_jacobian_score += jacobian_score[file_id]
    
    print("====================================================================================")
    print("- Your Total Score : {:00.03f} / {:00.03f}".format(
        total_fk_score + total_jacobian_score, FK_SCORE_MAX + JACOBIAN_SCORE_MAX))
    print("====================================================================================")

def main(args):

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet env without GUI
    visualize = args.gui
    physics_client_id = p.connect(p.GUI if visualize else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=90,
        cameraPitch=0,
        cameraTargetPosition=[0.7, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env
    robot = ur5Env(physics_client_id, use_IK=1)

    # -------------------------------------------- #
    # --- Test your Forward Kinematic function --- #
    # -------------------------------------------- #

    testcase_files = [
        'test_case/fk_test_case_easy.json',
        'test_case/fk_test_case_medium.json',
        'test_case/fk_test_case_hard.json',
        # 'test_case/fk_test_case_ta1.json',
        # 'test_case/fk_test_case_ta2.json',
    ]

    # scoring your algorithm
    score_fk(robot, testcase_files, visualize=args.visualize_pose)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', '-g', action='store_true', default=False, help='gui : whether show the window')
    parser.add_argument('--visualize-pose', '-vp', action='store_true', default=False, help='whether show the poses of end effector')
    args = parser.parse_args()
    main(args)
