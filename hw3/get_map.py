import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import cv2


def plot_align_point(x, y, z, tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01):
    _0255 = np.vstack((tmp_pcd_color_0255, np.array([0, 0, 0], dtype=np.uint8)))
    _01 = np.vstack((tmp_pcd_color_01, np.array([0, 0, 0], dtype=np.float32)))
    _coor = np.vstack((tmp_pcd_coor, np.array([x, y, z], dtype=np.float32)))

    plt.figure()
    plt.scatter(_coor[:, 2], _coor[:, 0], s=0.2, c=_01)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('tmp_map.png', dpi=300, bbox_inches='tight', pad_inches=0)
    
    correspond_color_0255 = np.array([0, 0, 0], dtype=np.uint8)
    tmp_img = cv2.imread('tmp_map.png')
    print(tmp_img.shape)
    correspond_index = np.where(np.all(tmp_img == correspond_color_0255, axis=-1))
    correspond_index = np.column_stack([correspond_index[0], correspond_index[1]])
    avg_correspond_index = np.mean(correspond_index, axis=0)
    return avg_correspond_index
    
    

if __name__ == '__main__':
    # load point cloud
    pcd_coor = np.load('semantic_3d_pointcloud/point.npy')
    pcd_color_01 = np.load('semantic_3d_pointcloud/color01.npy')
    pcd_color_0255 = np.load('semantic_3d_pointcloud/color0255.npy')
    
    # Remove roof or floor by y value
    roof_index = pcd_coor[:, 1] > -0.001
    floor_index = pcd_coor[:, 1] < -0.03
    other_index = ~(roof_index | floor_index)
    
    # plot point cloud w/ roof remove only
    plt.figure()
    plt.scatter(pcd_coor[floor_index, 2], pcd_coor[floor_index, 0], s=0.2, c=pcd_color_01[floor_index])
    plt.scatter(pcd_coor[other_index, 2], pcd_coor[other_index, 0], s=0.2, c=pcd_color_01[other_index])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('map_rm_roof.png', dpi=300, bbox_inches='tight', pad_inches=0)
    
    # plot point cloud w/ both removw
    plt.figure()
    plt.scatter(pcd_coor[other_index, 2], pcd_coor[other_index, 0], s=0.2, c=pcd_color_01[other_index])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('map_rm_both.png', dpi=300, bbox_inches='tight', pad_inches=0)
    
    # append color [0,0,0] to pcd_color_0255
    tmp_pcd_color_0255 = pcd_color_0255[other_index]
    tmp_pcd_color_01 = pcd_color_01[other_index]
    tmp_pcd_coor = pcd_coor[other_index]
    
    p1 = np.array([0, 0, 0], dtype=np.float32)
    p2 = np.array([0.05, 0, 0.2], dtype=np.float32)
    p1_2d = plot_align_point(p1[0], p1[1], p1[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    p2_2d = plot_align_point(p2[0], p2[1], p2[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    
    source = np.float32([[p1[0], p1[2]], [p2[0], p2[2]]])
    dest = np.float32([[p1_2d[0], p1_2d[1]], [p2_2d[0], p2_2d[1]]])
    transformation_matrix, _ = cv2.estimateAffinePartial2D(source, dest)
    # use this: np.matmul([0, 0, 1], transformation_matrix) # make sure turn source point to homogeneous coordinate
    breakpoint()
    
    
def get_coordinate_transform_matrix():
    # Get transformation matrix from semantic map to point cloud
    pcd_coor = np.load('semantic_3d_pointcloud/point.npy')
    pcd_color_01 = np.load('semantic_3d_pointcloud/color01.npy')
    pcd_color_0255 = np.load('semantic_3d_pointcloud/color0255.npy')
    
    roof_index = pcd_coor[:, 1] > -0.001
    floor_index = pcd_coor[:, 1] < -0.03
    other_index = ~(roof_index | floor_index)
    
    tmp_pcd_color_0255 = pcd_color_0255[other_index]
    tmp_pcd_color_01 = pcd_color_01[other_index]
    tmp_pcd_coor = pcd_coor[other_index]
    
    p1 = np.array([0, 0, 0.15], dtype=np.float32) 
    p2 = np.array([0.05, 0, 0.2], dtype=np.float32)
    p3 = np.array([0.05, 0, 0.15], dtype=np.float32)
    p4 = np.array([-0.05, 0, -0.1], dtype=np.float32)
    
    p1_2d = plot_align_point(p1[0], p1[1], p1[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    p2_2d = plot_align_point(p2[0], p2[1], p2[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    p3_2d = plot_align_point(p3[0], p3[1], p3[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    p4_2d = plot_align_point(p4[0], p4[1], p4[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    
    print(f"{p1_2d} -> {p1}")
    print(f"{p2_2d} -> {p2}")
    print(f"{p3_2d} -> {p3}")
    print(f"{p4_2d} -> {p4}")
    
    dest = np.float32([[p1[0], p1[2]], [p2[0], p2[2]], [p3[0], p3[2]], [p4[0], p4[2]]])
    source = np.float32([[p1_2d[0], p1_2d[1]], [p2_2d[0], p2_2d[1]], [p3_2d[0], p3_2d[1]], [p4_2d[0], p4_2d[1]]])
    
    
    transformation_matrix, _ = cv2.findHomography(source, dest)
    transformation_matrix = transformation_matrix * 10000 / 255.0 # convert from pointcloud to habitat
    return transformation_matrix