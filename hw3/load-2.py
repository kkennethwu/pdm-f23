import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
from get_map import get_coordinate_transform_matrix
from RRT import get_path
import time
from scipy.spatial.transform import Rotation as R
import math
import os
from PIL import Image
from get_semantic_id import get_semantic_id
from save_to_gif import save_to_gif

turn_deg = 2.0
move_forward_length = 0.05
# 1. get coordinate transform matrix from pixel to habitat
trans = get_coordinate_transform_matrix()
print("transformation matrix: ", trans)
# 2. get path by RRT
path, target_point, target_object, target_color = get_path()
target_semantic_id = get_semantic_id(target_object)

print("path: ", path)
print("target_point: ", target_point)
_3D_path = np.matmul(np.hstack((path, np.ones((path.shape[0], 1)))), trans.T) # x, z
_3D_target_point = np.matmul(np.append(target_point, 1), trans.T) # x, z
print("3D starting point: ", _3D_path[0])
print("3D target point: ", _3D_target_point)


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
path = "replica_v1/apartment_0/habitat/info_semantic.json"

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

######

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle             ###
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=move_forward_length) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_deg) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_deg)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space
agent_state.position = np.array([_3D_path[-1][0], 0.0, _3D_path[-1][1]]) # starting point
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")


def navigateAndSee(action="", frame=None):
    if action in action_names:
        observations = sim.step(action)
        
        if frame % 5 == 0:
            semantic_id = id_to_label[observations["semantic_sensor"]]
            target_id_region = np.where(semantic_id == target_semantic_id)  
            semamtic_img = transform_semantic(id_to_label[observations["semantic_sensor"]])
            
            
            rgb_img = transform_rgb_bgr(observations["color_sensor"])
            # add a transparent mask to highlight the target object
            red = np.full(rgb_img.shape, (0, 0, 255), dtype=np.uint8)
            if len(target_id_region[0]) > 0:
                blend_img = cv2.addWeighted(rgb_img, 0.5, red, 0.5, 0)
                rgb_img[target_id_region] = blend_img[target_id_region]
                
                
            
            cv2.imwrite(f"path_{target_object}/semantic_{frame}.jpg", semamtic_img)
            cv2.imwrite(f"path_{target_object}/RGB_{frame}.jpg", rgb_img)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        # print("camera pose: x y z rw rx ry rz")
        # print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)





def rotate_camera_to_direction(direction, i): # direction [x, z]
    sensor_state = agent.get_state().sensor_states['color_sensor']
    current_yaw = math.atan2(2.0 * (sensor_state.rotation.w * sensor_state.rotation.y + 
                                    sensor_state.rotation.x * sensor_state.rotation.z), 
                             1.0 - 2.0 * (sensor_state.rotation.y**2 + sensor_state.rotation.z**2))
    desired_yaw = -math.atan2(direction[0], -direction[1])
    
    rotation_diff = desired_yaw - current_yaw
    # Normalize the rotation difference to be within -pi to pi range
    
    print("current yaw: ", np.degrees(current_yaw))
    print("desired yaw: ", np.degrees(desired_yaw))
    print("rotation_diff: ", np.degrees(rotation_diff))
    # breakpoint()
    if rotation_diff > math.pi:
        rotation_diff -= 2 * math.pi
    elif rotation_diff < -math.pi:
        rotation_diff += 2 * math.pi
    rotation_diff_deg = np.degrees(rotation_diff)

    turn_action = "turn_left" if rotation_diff_deg > 0 else "turn_right"
    return turn_action, abs(rotation_diff_deg)


if not os.path.exists(f"path_{target_object}"):
    os.mkdir(f"path_{target_object}")
i = len(_3D_path)-1
frame = 0
while i > 0:
    sensor_state = agent.get_state().sensor_states['color_sensor']
    print(f"##### {len(_3D_path)-1-i}th Node #####")
    print("sensor_state.position: ", sensor_state.position)
    print("current path point: ", _3D_path[i])
    print("next path point: ", _3D_path[i-1])
    direction = np.array([_3D_path[i-1][0] - sensor_state.position[0], _3D_path[i-1][1] - sensor_state.position[2]]) # [x, z]
    dir_length = np.linalg.norm(direction)
    direction /= dir_length
    
    turn_action, rotation_diff_deg = rotate_camera_to_direction(direction, i)
    
    print(f"{turn_action}: {rotation_diff_deg} degree")
    while(rotation_diff_deg > 0):
        navigateAndSee(turn_action, frame=frame)
        rotation_diff_deg -= turn_deg
        frame+=1
    print(f"move_forward: {dir_length}")
    while(dir_length > 0):
        navigateAndSee("move_forward", frame=frame)
        dir_length -= move_forward_length
        frame+=1
    i -= 1
print(f"##### Toward the target #####")
sensor_state = agent.get_state().sensor_states['color_sensor']
direction = np.array([_3D_target_point[0] - sensor_state.position[0], _3D_target_point[1] - sensor_state.position[2]]) # [x, z]
dir_length = np.linalg.norm(direction)
direction /= dir_length
turn_action, rotation_diff_deg = rotate_camera_to_direction(direction, i=-1)
print(f"{turn_action}: {rotation_diff_deg} degree")
while(rotation_diff_deg > 0):
    navigateAndSee(turn_action, frame=frame)
    rotation_diff_deg -= turn_deg
    frame+=1

save_to_gif(object=target_object)