import numpy as np
import open3d as o3d
import argparse
import os
import cv2

data_root = ""
seg_root = ""

def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000)
    )

    # raise NotImplementedError
    return result.transformation

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

def depth_image_to_point_cloud(rgb, depth, sem, z_threshold):
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

def depth_image_to_point_cloud2(rgb, depth, sem, z_threshold):
    # TODO: Get point cloud from rgb and depth image 
    H, W, focal, depth_scale = 512, 512, 256, 1000
    v, u = np.mgrid[0:H, 0:W]
    pcd = o3d.geometry.PointCloud()
    pcd_sem = o3d.geometry.PointCloud()

    z = depth[:, :, 0].astype(np.float32) / 255 * (-10)  #convert depth map to meters
    x = (u - W*.5) * z / focal
    y = (v - H*.5) * z / focal

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rgbs = (rgb[:, :, [2, 1, 0]].astype(np.float32) / 255).reshape(-1, 3)
    sem_colors = (sem[:, :, [2, 1, 0]].astype(np.float32) / 255).reshape(-1, 3)

    valid = points[:, 2] <= z_threshold
    points = points[valid, :]
    rgbs = rgbs[valid, :]
    sem_colors = sem_colors[valid, :]

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    pcd_sem.points = o3d.utility.Vector3dVector(points)
    pcd_sem.colors = o3d.utility.Vector3dVector(sem_colors)
    
    # raise NotImplementedError
    return pcd, pcd_sem


def get_point_cloud_list(args, z_threshold):
    point_cloud_list = []
    point_cloud_list_sem = []
    item_len = len(os.listdir(f"{args.data_root}/depth"))

    for i in range(1, item_len+1):
        depth = cv2.imread(f"{args.data_root}/depth/{i}.png")
        rgb = cv2.imread(f"{args.data_root}/rgb/{i}.png")
        seg = cv2.imread(f"{args.seg_root}/apartment_{i-1:06d}.png")
        if args.seg_gt: # use gt segmentation
            seg = seg[:, 512:1024, :]
        else: # use predicted segmentation
            seg = seg[:, 1024:, :]
        
        # pcd_rgb = depth_image_to_point_cloud(rgb, depth, seg, z_threshold)
        
        pcd, pcd_sem = depth_image_to_point_cloud2(rgb, depth, seg, z_threshold)
        
        point_cloud_list.append(pcd)
        point_cloud_list_sem.append(pcd_sem)
    
    return point_cloud_list, point_cloud_list_sem

def custom_voxel_down1(pcd, voxel_size):
    # TODO: Implement custom voxel down-sampling function
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    voxel_labels = []
    voxel_centers = []
    voxel_size_half = voxel_size / 2

    for i, point in enumerate(pcd.points):
        voxel_center = np.floor(point / voxel_size) * voxel_size + voxel_size_half
        if voxel_center.tolist() in voxel_centers:
            continue
        [k, idx, _] = pcd_tree.search_radius_vector_3d(voxel_center, voxel_size_half)
        labels = np.asarray(pcd.colors)[idx]
        unique_labels, counts = np.unique(labels, axis=0, return_counts=True)
        if (counts is not None) and (len(counts) > 0):
            majority_label = unique_labels[np.argmax(counts)]
        else:
            majority_label = np.asarray([0, 0, 0])
        voxel_labels.append(majority_label)
        voxel_centers.append(voxel_center.tolist())

    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    down_pcd.colors = o3d.utility.Vector3dVector(voxel_labels)
    # raise NotImplementedError
    return down_pcd

def custom_voxel_down2(pcd, voxel_size):
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Compute voxel centers and labels
    voxel_centers = []
    voxel_labels = []

    voxels = voxel_grid.get_voxels()
    for i in range(len(voxels)):
        voxel = voxels[i]
        voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
        [k, idx, _] = pcd_tree.search_radius_vector_3d(voxel_center, voxel_size / 2)
        if len(idx) == 0:
            continue
        
        labels = np.asarray(pcd.colors)[idx]
        unique_labels, counts = np.unique(labels, axis=0, return_counts=True)
        if (counts is not None) and (len(counts) > 0):
            majority_label = unique_labels[np.argmax(counts)]
        else:
            majority_label = np.asarray([0, 0, 0])
        voxel_centers.append(voxel_center)
        voxel_labels.append(majority_label)

    # Create down-sampled point cloud
    down_pcd = o3d.geometry.PointCloud()
    down_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    down_pcd.colors = o3d.utility.Vector3dVector(voxel_labels)
    return down_pcd

def custom_voxel_down(pcd, pcd_sem, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_sem)
    
    # Compute voxel centers and labels
    voxel_labels = []
    for i in range(len(pcd_down.points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd_down.points[i], voxel_size / 2)
        if len(idx) == 0:
            continue
        
        labels = np.asarray(pcd_sem.colors)[idx]
        unique_labels, counts = np.unique(labels, axis=0, return_counts=True)
        if (counts is not None) and (len(counts) > 0):
            majority_label = unique_labels[np.argmax(counts)]
        else:
            majority_label = np.asarray([0, 0, 0])
        voxel_labels.append(majority_label)
    pcd_down.colors = o3d.utility.Vector3dVector(voxel_labels)  
    
    return pcd_down

def preprocess_point_cloud(pcd, pcd_sem, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = custom_voxel_down(pcd, pcd_sem, voxel_size) 

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, 
                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # raise NotImplementedError
    return pcd_down, pcd_fpfh

def preprocess(point_cloud_list, pcd_list_sem, voxel_size):
    pcd_down_list = []
    pcd_feature_list = []
    for i, (pcd, pcd_sem) in enumerate(zip(point_cloud_list, pcd_list_sem)):
        print("preprocess pcd_", i)
        pcd_down, pcd_feature = preprocess_point_cloud(pcd, pcd_sem, voxel_size)
        pcd_down_list.append(pcd_down)
        pcd_feature_list.append(pcd_feature)
    return pcd_down_list, pcd_feature_list

# def custom_voxel_down(pcd, voxel_size):
#     #TODO: implement your own voxel down
#     raise NotImplementedError

def reconstruct(args):
    #TODO: reconstruct the 3d semantic map
    
    # init pred_cam_pos_list
    pred_cam_pose = [np.identity(4)]
    # get point cloud list
    z_threshold = 80 / 255 * 10
    pcd_list, pcd_list_sem = get_point_cloud_list(args, z_threshold=z_threshold)
    # preprocess
    voxel_size = 0.07
    pcd_down_list, pcd_fpfh_list = preprocess(pcd_list, pcd_list_sem, voxel_size=voxel_size)
    
    # ICP
    for i in range(1, len(pcd_list)):
        print(i)
        pcd_down_source, pcd_down_target = pcd_down_list[i], pcd_down_list[i-1]
        pcd_fpfh_source, pcd_fpfh_target = pcd_fpfh_list[i], pcd_fpfh_list[i-1]
        # global registration
        init_trans = execute_global_registration(pcd_down_source, pcd_down_target, 
                                                 pcd_fpfh_source, pcd_fpfh_target, voxel_size=voxel_size)
        # local registration
        trans = local_icp_algorithm(pcd_down_source, pcd_down_target, init_trans, threshold=voxel_size*0.4)
        # elif args.version == 'my_icp':
        #     trans = my_local_icp_algorithm(pcd_down_source, pcd_down_target, init_trans, 
        #                                    voxel_size=voxel_size, max_iters=100, convergence_threshold=1e-6)
        
        pred_cam_pose.append(pred_cam_pose[i-1] @ trans)
    
    for i in range(len(pcd_list_sem)):
        pcd_list_sem[i].transform(pred_cam_pose[i])

    return pcd_list_sem, np.array(pred_cam_pose)
    
    
    raise NotImplementedError

def visualize(result_pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for i, pcd in enumerate(result_pcd):
        vis.add_geometry(pcd)
    
    # # Wait for the visualization window to be fully rendered
    vis.update_renderer()
    # Capture the top view as an image
    vis.capture_screen_image("top_view.png", do_render=True)
    
    # Close the visualization window
    vis.destroy_window()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1, choices=[1,2])
    parser.add_argument('-d', '--dataset', type=int, default=1, choices=[1,2])
    parser.add_argument('--data_root', type=str, default='../hw1/data_collection/')
    parser.add_argument('--seg_root', type=str, default='./data_collection/')
    parser.add_argument('--seg_gt', action='store_true', help='use gt segmentation')
    args = parser.parse_args()
    
    if args.floor == 1:
        args.data_root = args.data_root + 'first_floor/'
    elif args.floor == 2:
        args.data_root = args.data_root + 'second_floor/'
    args.seg_root = args.seg_root + f'dataset{args.dataset}/' + f'floor{args.floor}/'
    
    # Output result point cloud and estimated camera pose
    result_pcd, pred_cam_pose = reconstruct(args)
    
    # Rmove roof
    for i in range(len(result_pcd)):
        points = np.asarray(result_pcd[i].points)
        rgbs = np.asarray(result_pcd[i].colors)
        valid = (points[:, 1] <= 0.25)
        points, rgbs = points[valid], rgbs[valid]

        result_pcd[i].points = o3d.utility.Vector3dVector(points[:, 0:3])
        result_pcd[i].colors = o3d.utility.Vector3dVector(rgbs[:, 0:3])
    
    # Rotate to BEV
    rot_bev1 = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
    rot_bev2 = np.array([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    
    for i in range(len(result_pcd)):
        result_pcd[i].transform(rot_bev1)
        result_pcd[i].transform(rot_bev2)
    
    # visualize
    # o3d.visualization.draw_geometries(result_pcd)
    
    visualize(result_pcd)
    
    
    

    