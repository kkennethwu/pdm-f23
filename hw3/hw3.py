import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # load point cloud
    pcd_coor = np.load('semantic_3d_pointcloud/point.npy')
    pcd_color_01 = np.load('semantic_3d_pointcloud/color01.npy')
    pcd_color_0255 = np.load('semantic_3d_pointcloud/color0255.npy')

    # remove roof or floor
    roof_index = pcd_coor[:, 1] > 0.0
    floor_index = pcd_coor[:, 1] < -0.03
    other_index = ~(roof_index | floor_index)
    
    # plot point cloud w/ roof remove only
    plt.figure()
    plt.scatter(pcd_coor[floor_index, 2], pcd_coor[floor_index, 0], s=1, c=pcd_color_01[floor_index])
    plt.scatter(pcd_coor[other_index, 2], pcd_coor[other_index, 0], s=1, c=pcd_color_01[other_index])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('map_rm_roof.png', dpi=300, bbox_inches='tight', pad_inches=0)
    
    # plot point cloud w/ both removw
    plt.figure()
    plt.scatter(pcd_coor[other_index, 2], pcd_coor[other_index, 0], s=1, c=pcd_color_01[other_index])
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('map_rm_both.png', dpi=300, bbox_inches='tight', pad_inches=0)
    
    