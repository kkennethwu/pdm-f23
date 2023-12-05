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
    correspond_index = np.where(np.all(tmp_img == correspond_color_0255, axis=-1))
    correspond_index = np.column_stack([correspond_index[0], correspond_index[1]])
    avg_correspond_index = np.mean(correspond_index, axis=0)
    return avg_correspond_index
    
    

if __name__ == '__main__':
    # load point cloud
    pcd_coor = np.load('semantic_3d_pointcloud/point.npy')
    pcd_color_01 = np.load('semantic_3d_pointcloud/color01.npy')
    pcd_color_0255 = np.load('semantic_3d_pointcloud/color0255.npy')

    
    
    

    # Remove roof or floor
    # remove by color
    # floor_color = np.array([255, 194, 7], dtype=np.uint8)
    # floor_index = np.all(pcd_color_0255 == floor_color, axis=-1)
    # roof_color = np.array([8, 255, 214], dtype=np.uint8)
    # roof_index = np.all(pcd_color_0255 == roof_color, axis=-1)
    # remove by height
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
    
    # find the coordinate relation between pointcloud and semantic map
    # correspond_color_0255 = np.array([7, 7, 7], dtype=np.uint8)
    # correspond_color_01 = correspond_color_0255 / 255.0
    # pcd_color_01[0] = correspond_color_01 # change point [0,0,0] to color with [7,7,7]
    
    # # plot map w/ nothing remove
    # plt.figure()
    # plt.scatter(pcd_coor[:, 2], pcd_coor[:, 0], s=0.2, c=pcd_color_01)
    # plt.axis('off')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig('map_rm_no.png', dpi=300, bbox_inches='tight', pad_inches=0)
    
    # tmp_img = cv2.imread('map_rm_no.png')
    # correspond_index = np.where(np.all(tmp_img == correspond_color_0255, axis=-1))
    # breakpoint()
    
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
    
    p1 = np.array([-0.05, 0, 0.02], dtype=np.float32) 
    p2 = np.array([0.05, 0, 0.2], dtype=np.float32)
    
    p2_2d = plot_align_point(p2[0], p2[1], p2[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    p1_2d = plot_align_point(p1[0], p1[1], p1[2], tmp_pcd_coor, tmp_pcd_color_0255, tmp_pcd_color_01)
    
    
    dest = np.float32([[p1[0], p1[2]], [p2[0], p2[2]]])
    source = np.float32([[p1_2d[0], p1_2d[1]], [p2_2d[0], p2_2d[1]]])
    transformation_matrix, _ = cv2.estimateAffinePartial2D(source, dest) # convert from pixel to pointcloud
    transformation_matrix = transformation_matrix * 10000 / 255.0 # convert from pointcloud to habitat
    return transformation_matrix