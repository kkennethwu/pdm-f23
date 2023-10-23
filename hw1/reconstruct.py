import numpy as np
import open3d as o3d
import argparse
import cv2
import os
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from random import sample
import time




def depth_image_to_point_cloud(rgb, depth, z_threshold):
    # TODO: Get point cloud from rgb and depth image 
    H, W, focal, depth_scale = 512, 512, 256, 1000
    v, u = np.mgrid[0:H, 0:W]
    pcd = o3d.geometry.PointCloud()

    z = depth[:, :, 0].astype(np.float32) / 255 * (-10)  #convert depth map to meters
    x = (u - W*.5) * z / focal
    y = (v - H*.5) * z / focal
    # pcd_array = np.concatenate([np.dstack((x, y, z)), rgb[:, :, ::-1]], 2)
    # pcd_array.reshape(-1, 6)

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rgbs = (rgb[:, :, [2, 1, 0]].astype(np.float32) / 255).reshape(-1, 3)


    valid = points[:, 2] <= z_threshold
    points = points[valid, :]
    rgbs = rgbs[valid, :]

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    # raise NotImplementedError
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, 
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # raise NotImplementedError
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, 
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    # raise NotImplementedError
    return result.transformation

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        )
    )
    return result.transformation

def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000)
    )

    # raise NotImplementedError
    return result.transformation


def nearest_neighbor(source, target):
    tree = cKDTree(target)
    distances, indices = tree.query(source, k=1)

    return distances, indices


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size, max_iters, convergence_threshold):
    # TODO: Write your own ICP function
    trans = trans_init.copy()
    trans_update = trans_init
    # source_down = source_down.transform(trans)
    # sample from normal space
    source_normal = np.asarray(source_down.normals)
    target_noraml = np.asarray(target_down.normals)
    num_sample_points = min(len(source_normal), len(target_noraml))
    source_sampled_indices = sample(range(len(source_normal)), num_sample_points)
    target_sampled_indices = sample(range(len(target_noraml)), num_sample_points)    
    # sampled_points
    source_points = np.asarray(source_down.points)[source_sampled_indices]
    target_points = np.asarray(target_down.points)[target_sampled_indices]
    source_normal = source_normal[source_sampled_indices]
    
    #
    # Transform the source with current transformation matrix
    source_points_homo = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    source_points = np.dot(source_points_homo, trans_update.T)[:, :3]
    
    # Find data association (corresponding points)
    distances, indices = nearest_neighbor(source_points, target_points)
    
    
    for iter in range(max_iters):
        error = np.mean((source_points - target_points[indices])**2) # want the error to be minimized
        print("error: ", error)
        
        # compute the optimal solution of the transformation from estimated data association
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points[indices], axis=0)
        source_center_diff = source_points - source_center # p' = {p - mean}
        target_center_diff = target_points[indices] - target_center # x' = {x - x_mean} 
        W = np.matmul(target_center_diff.T, source_center_diff)
        U, _, VT = np.linalg.svd(W)
        optimal_rotation = U @ VT
        if np.linalg.det(optimal_rotation) < 0: # reflection case
            VT[:, :] *= -1
            optimal_rotation = U @ VT

        optimal_translation = target_center - np.dot(source_center, optimal_rotation.T)
        
        trans_update = np.eye(4)
        trans_update[:3, :3] = optimal_rotation
        trans_update[:3, 3] = optimal_translation
        # Update the current transformation
        trans = trans_update @ trans       
        
        # Check for convergence 
        if np.linalg.norm(trans_update - np.identity(4)) < convergence_threshold:
            break 
        
        # Transform the source with current transformation matrix
        source_points_homo = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
        source_points = np.dot(source_points_homo, trans_update.T)[:, :3]
    print("final iter: ", iter)
    
    return trans_init

def point2plane_error(trans, source_points, target_points, target_normal):
    rotation = trans[:3, :3]
    translation = trans[:3, 3]
    transformed_source = np.dot(source_points, rotation.T) + translation
    
    error = np.sum(np.sum((target_points - source_points)* target_normal, axis=1)**2)
    return error
        
def my_local_icp_algorithm_point2plane(source_down, target_down, trans_init, voxel_size, max_iters, convergence_threshold):
    # TODO: Write your own ICP function
    trans = trans_init.copy()
    # sample from normal space
    source_normal = np.asarray(source_down.normals)
    target_normal = np.asarray(target_down.normals)
    num_sample_points = min(len(source_normal), len(target_normal))
    source_sampled_indices = sample(range(len(source_normal)), num_sample_points)
    target_sampled_indices = sample(range(len(target_normal)), num_sample_points)
    # sampled_points
    source_points = np.asarray(source_down.points)[source_sampled_indices]
    target_points = np.asarray(target_down.points)[target_sampled_indices]
    source_normal = source_normal[source_sampled_indices]
    target_normal = target_normal[target_sampled_indices]
    
    for iter in range(max_iters):
        # # Transform the source with current transformation matrix
        # source_points_homo = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
        # source_points = np.dot(source_points_homo, trans.T)[:, :3]
        
        
        # Find data association (corresponding points)
        distances, indices = nearest_neighbor(source_points, target_points)
        # error = np.sum(np.sum((target_points[indices] - source_points)* target_normal[indices], axis=1)**2)
        # print("error: ", error)
        # breakpoint()
        # compute the optimal solution of the transformation from estimated data association
        initial_guess = np.identity(3)
        initial_guess = np.append(initial_guess, [0, 0, 0])
        result = minimize(point2plane_error, initial_guess, args=(source_points, target_points[indices], target_normal[indices]), method='trust-ncg')
        breakpoint()
            
            
        
        
        
        #
        
        
        
        U, _, VT = np.linalg.svd(A_sum)
        optimal_rotation = U @ VT
        optimal_translation = np.dot(b_sum, U)
        
        trans_update = np.eye(4)
        trans_update[:3, :3] = optimal_rotation
        trans_update[:3, 3] = optimal_translation
        # Update the current transformation
        trans = trans_update @ trans       
        
        # Check for convergence 
        if np.linalg.norm(trans_update - np.identity(4)) < convergence_threshold:
            break 
        # breakpoint()
    print("final iter: ", iter)
    
    return trans
    


def get_point_cloud_list(args, z_threshold):
    point_cloud_list = []
    item_len = len(os.listdir(f"{args.data_root}/depth"))

    for i in range(1, item_len+1):
        depth = cv2.imread(f"{args.data_root}/depth/{i}.png")
        rgb = cv2.imread(f"{args.data_root}/rgb/{i}.png")
        
        pcd = depth_image_to_point_cloud(rgb, depth, z_threshold)
        point_cloud_list.append(pcd)
    
    return point_cloud_list

def preprocess(point_cloud_list, voxel_size):
    pcd_down_list = []
    pcd_feature_list = []
    for pcd in point_cloud_list:
        pcd_down, pcd_feature = preprocess_point_cloud(pcd, voxel_size)
        pcd_down_list.append(pcd_down)
        pcd_feature_list.append(pcd_feature)
    return pcd_down_list, pcd_feature_list

def reconstruct(args):
    # TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    # init pred_cam_pos_list
    pred_cam_pose = [np.identity(4)]
    # get point cloud list
    z_threshold = 80 / 255 * 10
    pcd_list = get_point_cloud_list(args, z_threshold=z_threshold)
    # preprocess 
    voxel_size = 0.07
    pcd_down_list, pcd_fpfh_list = preprocess(pcd_list, voxel_size=voxel_size)
    # 
    # breakpoint()
    for i in range(1, len(pcd_list)):
        print(i)
        pcd_down_source, pcd_down_target = pcd_down_list[i], pcd_down_list[i-1]
        pcd_fpfh_source, pcd_fpfh_target = pcd_fpfh_list[i], pcd_fpfh_list[i-1]
        # global registration
        init_trans = execute_global_registration(pcd_down_source, pcd_down_target, 
                                                 pcd_fpfh_source, pcd_fpfh_target, voxel_size=voxel_size)
        # init_trans = execute_fast_global_registration(pcd_down_source, pcd_down_target, 
        #                                               pcd_fpfh_source, pcd_fpfh_target, voxel_size=voxel_size)
        # local registration
        if args.version == 'open3d':
            trans = local_icp_algorithm(pcd_down_source, pcd_down_target, init_trans, threshold=voxel_size*0.4)
        elif args.version == 'my_icp':
            trans = my_local_icp_algorithm(pcd_down_source, pcd_down_target, init_trans, 
                                           voxel_size=voxel_size, max_iters=100, convergence_threshold=1e-6)
        
        pred_cam_pose.append(pred_cam_pose[i-1] @ trans)
    
    for i in range(len(pcd_list)):
        pcd_list[i].transform(pred_cam_pose[i])

    return pcd_list, np.array(pred_cam_pose)


if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pose = reconstruct(args)
    # rmove roof
    for i in range(len(result_pcd)):
        points = np.asarray(result_pcd[i].points)
        rgbs = np.asarray(result_pcd[i].colors)
        valid = (points[:, 1] <= 0.25)
        points, rgbs = points[valid], rgbs[valid]

        result_pcd[i].points = o3d.utility.Vector3dVector(points[:, 0:3])
        result_pcd[i].colors = o3d.utility.Vector3dVector(rgbs[:, 0:3])
        


    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    gt_cam_pose_7d = np.load(f'{args.data_root}/GT_pose.npy')
    gt_cam_pose = np.tile(np.eye(4), (gt_cam_pose_7d.shape[0], 1, 1))
    gt_cam_pose[:, 0:3, 0:3] = Rotation.from_quat(gt_cam_pose_7d[:, 3:7]).as_matrix()
    gt_cam_pose[:, 0:3, 3] = gt_cam_pose_7d[:, 0:3]
    # breakpoint()
    gt_cam_pose = np.tile(np.linalg.inv(gt_cam_pose[[0]]), (gt_cam_pose.shape[0], 1, 1)) @ gt_cam_pose
    gt_cam_position = gt_cam_pose[:, 0:3, 3]
    gt_cam_position[:, 0] = -gt_cam_position[:, 0]
    gt_cam_position[:, 2] = -gt_cam_position[:, 2]
    pred_cam_position = pred_cam_pose[:, 0:3, 3]

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    
    print("Mean L2 distance: ", np.mean(np.linalg.norm(gt_cam_pose - pred_cam_pose)))
    # breakpoint()
    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    # breakpoint()
    edges = [[i, i+1] for i in range(gt_cam_pose.shape[0] - 1)]
    gt_color = [[0, 0, 0] for i in range(len(edges))]
    pred_color = [[1, 0, 0] for i in range(len(edges))]

    gt_line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(gt_cam_position),
        lines = o3d.utility.Vector2iVector(edges)
    )
    gt_line_set.colors = o3d.utility.Vector3dVector(gt_color)

    pred_line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(pred_cam_position),
        lines = o3d.utility.Vector2iVector(edges)
    )
    pred_line_set.colors = o3d.utility.Vector3dVector(pred_color)


    o3d.visualization.draw_geometries(result_pcd+[gt_line_set, pred_line_set])

    # o3d.visualization.draw_geometries()
