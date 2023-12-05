import scipy.io
import pandas as pd
import cv2
from scipy.spatial import KDTree
import numpy as np

start_point = []


def click_event(event, x, y, flags, param):
    global start_point, img, window_size
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        start_point = [x, y]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Smaller Image', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, y)
        start_point = [x, y]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Smaller Image', img)

def sample_integer_point(nearest_point, new_point, sample_num=20):
    nearest_point = np.array(nearest_point).astype(int)
    new_point = np.array(new_point).astype(int)
    line = np.linspace(nearest_point, new_point, sample_num, endpoint=True)
    line = line.astype(int)
    return line

class RRTNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
      
class RRT:
    def __init__(self, start_point, target_color, sematic_map, step_size=5, max_iter=100):
        # start_point, targt_point, goal_point, k vertices
        self.start_point = start_point
        self.target_color = target_color
        self.avg_target_point = None
        
        self.start_Node = RRTNode(start_point[0], start_point[1])
        self.map = sematic_map
        self.MAX_step_size = step_size
        self.MAX_iter = max_iter
        
        self.data = [self.start_point]
        self.kd_tree = KDTree(self.data) # For fast searching 
        self.tree = [self.start_Node] # For indexing
        
    
    def get_random_point(self):
        return [np.random.randint(0, self.map.shape[0]), np.random.randint(0, self.map.shape[1])]
    
    def find_nearest_node(self, random_point):
        _, idx = self.kd_tree.query(random_point)
        return self.kd_tree.data[idx], self.tree[idx]
    
    def check_collision(self, nearest_point, new_point):
        # TODO: check collision
        line = sample_integer_point(nearest_point, new_point, sample_num=20)
        for p in line:
            if (self.map[p[0], p[1]] != [255, 255, 255]).any():
                return True
        return False
    
    def extend(self, nearest_point, random_point):
        dir = np.array(random_point) - np.array(nearest_point)
        dir = dir / np.linalg.norm(dir)
        new_point = nearest_point + self.MAX_step_size * dir
        return new_point.astype(int)
    
    def check_target_color(self, new_point, last_point):
        # ## Method1
        # if np.linalg.norm(np.array(new_point) - np.array(self.avg_target_point)) < 100:
        #     return True
        # return False
        
        # ## Method2
        # # TODO: Check if target color is in front of new point
        # dir = np.array(new_point) - np.array(last_point)
        # is_inside = lambda point: 0 <= point[0] < self.map.shape[0] and 0 <= point[1] < self.map.shape[1]
        # line = sample_integer_point(new_point, new_point + dir, sample_num=20)
        # line = np.array([l for l in line if is_inside(l)])
        # region_around_point = self.map[line[:, 0], line[:, 1]]
        # # breakpoint()
        
        
        ## Method3
        # target_point = (new_point + dir / 2).astype(int)
        # region_around_point = self.map[target_point[0]-20:target_point[0]+21, target_point[1]-20:target_point[1]+21]
        region_around_point = self.map[new_point[0]-40:new_point[0]+41, new_point[1]-40:new_point[1]+41]
        # is_inside = lambda point: 0 <= point[0] < self.map.shape[0] and 0 <= point[1] < self.map.shape[1]
        
        # Check if any pixel in the region matches the target color
        is_target_color_region = np.all(region_around_point == self.target_color, axis=-1)
        if np.any(is_target_color_region):
            return True
        return False
    
    def run_algorithm(self):
        print("#### Start RRT path planning ####")
        print("Start point: ", self.start_point)
        print("Target color: ", self.target_color)
        
        for i in range(self.MAX_iter):
            # Random sample points
            random_point = self.get_random_point()
            # Find nearest node in tree
            nearest_point, nearest_Node = self.find_nearest_node(random_point)
            # Extend the tree towards random point
            new_point = self.extend(nearest_point, random_point)
            # Check collision, if collision, sample again
            # breakpoint()
            if self.check_collision(nearest_point, new_point):
                continue
            # Add new node to the tree
            # cv2.namedWindow('Smaller Image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Smaller Image', *window_size)
            # cv2.imshow('Smaller Image', img)
            # cv2.line(img, (nearest_Node.y, nearest_Node.x), (new_point[1], new_point[0]), (0, 0, 0), 3)
            # cv2.circle(img, (new_point[1], new_point[0]), 6, (147, 20, 255), thickness=-1)
            # cv2.imshow('Smaller Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            new_Node = RRTNode(new_point[0], new_point[1], parent=nearest_Node)
            self.tree.append(new_Node)
            self.data.append(new_point)
            self.kd_tree = KDTree(self.data) # update kd tree 
            
            # Check if target color is found
            if self.check_target_color(new_point, nearest_point):
                print("Target color found.")
                path = []
                while new_Node.parent is not None:
                    path.append([new_Node.x, new_Node.y])
                    new_Node = new_Node.parent
                path.append([new_Node.x, new_Node.y])
                return path, self.tree
            
        # Visulize the tree
        # tmp_map = self.map.copy()
        # cv2.namedWindow('Smaller Image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Smaller Image', *window_size)
        # cv2.imshow('Smaller Image', tmp_map)
        # for node in self.tree:
        #     if node.parent is not None:
        #         # print(node.x, node.y)
        #         cv2.line(tmp_map, (node.y, node.x), (node.parent.y, node.parent.x), (0, 0, 0), 5)
        #     cv2.circle(tmp_map, (node.y, node.x), 6, (147, 20, 255), thickness=-1)
        # cv2.imshow('Smaller Image', tmp_map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        print("Cannot find target color. Run RRT again.")
        return None, self.tree


def visualize(path, tree, img, avg_target_point):
    # cv2.namedWindow('Smaller Image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Smaller Image', *window_size)
    # cv2.imshow('Smaller Image', img)
    for node in tree:
        if node.parent is not None:
            cv2.line(img, (node.y, node.x), (node.parent.y, node.parent.x), (0, 0, 0), 5)
        cv2.circle(img, (node.y, node.x), 6, (147, 20, 255), thickness=-1)
    
    for i in range(0, len(path)-1):
        cv2.line(img, (path[i][1], path[i][0]), (path[i+1][1], path[i+1][0]), (0, 0, 255), 2)
        cv2.circle(img, (path[i][1], path[i][0]), 6, (0, 0, 255), thickness=-1)
    cv2.circle(img, (path[-1][1], path[-1][0]), 6, (0, 0, 255), thickness=-1)
    cv2.circle(img, (avg_target_point[1], avg_target_point[0]), 20, (0, 0, 255), thickness=-1)
    cv2.imwrite('path.png', img)
    
def get_path():
    df = pd.read_excel("color_coding_semantic_segmentation_classes.xlsx")
    names = df['Name'].tolist()
    color = scipy.io.loadmat('color101.mat')['colors']
    name_color_dict = dict(zip(names, color))

    # input target point
    while True:
        target_object = input('Target object: ')
        if name_color_dict.get(target_object) is None:
            print('Object not found.')
            continue
        target_color = name_color_dict[target_object]
        # convert target color from RGB to BGR
        target_color = target_color[::-1]
        break
    
    # click start point
    global start_point, img, window_size
    start_point = []
    img = cv2.imread('map_rm_both.png')
    original_img = img.copy()
    window_size = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.namedWindow('Smaller Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Smaller Image', *window_size)
    cv2.imshow('Smaller Image', img)
    cv2.setMouseCallback('Smaller Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    indices = np.where(np.all(img == target_color, axis=-1))
    pixel_coordinates = np.column_stack((indices[0], indices[1]))
    avg_pixel_coordinates = np.mean(pixel_coordinates, axis=0).astype(int)
    print(f"{target_object}: ", avg_pixel_coordinates)
    
    # RRT path planning
    path = None
    start_point = [start_point[1], start_point[0]]
    while path is None:
        rrt = RRT(start_point, target_color, original_img, step_size=100, max_iter=500)
        rrt.avg_target_point = avg_pixel_coordinates
        path, tree = rrt.run_algorithm()
    # Visulize the path and tree
    if path is not None:
        visualize(path, tree, original_img, avg_pixel_coordinates)
        return np.array(path), np.array(avg_pixel_coordinates)
    return None, None
    

if __name__ == '__main__':
    # Load data
    df = pd.read_excel("color_coding_semantic_segmentation_classes.xlsx")
    names = df['Name'].tolist()
    color = scipy.io.loadmat('color101.mat')['colors']
    name_color_dict = dict(zip(names, color))

    # input target point
    while True:
        target_object = input('Target object: ')
        if name_color_dict.get(target_object) is None:
            print('Object not found.')
            continue
        target_color = name_color_dict[target_object]
        # convert target color from RGB to BGR
        target_color = target_color[::-1]
        break
    
    # click start point
    start_point = []
    img = cv2.imread('map_rm_both.png')
    original_img = img.copy()
    window_size = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.namedWindow('Smaller Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Smaller Image', *window_size)
    cv2.imshow('Smaller Image', img)
    cv2.setMouseCallback('Smaller Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    indices = np.where(np.all(img == target_color, axis=-1))
    pixel_coordinates = np.column_stack((indices[0], indices[1]))
    avg_pixel_coordinates = np.mean(pixel_coordinates, axis=0).astype(int)
    print(f"{target_object}: ", avg_pixel_coordinates)
    
    
    # RRT path planning
    start_point = [start_point[1], start_point[0]]
    rrt = RRT(start_point, target_color, original_img, step_size=100, max_iter=500)
    rrt.avg_target_point = avg_pixel_coordinates
    path, tree = rrt.run_algorithm()
    
    # Visulize the path and tree
    if path is not None:
        visualize(path, tree, original_img, avg_pixel_coordinates)
    
        
    
    
    
    
    
    
    

