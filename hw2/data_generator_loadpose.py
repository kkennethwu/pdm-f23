import argparse
import json
import os
import numpy as np
from PIL import Image

import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim
from habitat_sim.agent import AgentState
from habitat_sim.utils.common import quat_from_angle_axis


def make_cfg(settings):
    sim_cfg = hsim.SimulatorConfiguration()
    if "enable_physics" in settings.keys():
        sim_cfg.enable_physics = settings["enable_physics"]
    else:
        sim_cfg.enable_physics = False
    if "physics_config_file" in settings.keys():
        sim_cfg.physics_config_file = settings["physics_config_file"]
    print("sim_cfg.physics_config_file = " + sim_cfg.physics_config_file)
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]

    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # create agent specifications
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }
    if sim_cfg.enable_physics:
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
            )
        }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def create_room(x_1, y_1, z_1, x_2, y_2, z_2):
    x_min = min(x_1, x_2)
    x_max = max(x_1, x_2)
    y_min = min(y_1, y_2)
    y_max = max(y_1, y_2)
    z_min = min(z_1, z_2)
    z_max = max(z_1, z_2)
    return {'x_min': x_min, 
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'z_min': z_min,
            'z_max': z_max}

class Generator:
    """Generator for replica dataset, rgb, depth, and semantics.
    """
    def __init__(self, path):
        self._dataset_path = os.path.normpath(path)
        self._scenes = ["apartment_0"]
        
        self._height = 512
        self._width = 512
        
        self._last_frame = None
        self._last_depth_frame = None
        self._last_semantic_frame = None
        
        self._scene_to_rooms = {
            "apartment_0": [
                create_room(-0.3, 1.4, 2.8, 1.68, 0.28, 3.04), # bedroom
                create_room(2.25, -1.38, 3.12, 3.52, -2.0, 2.19), # bathroom
                create_room(-1.65, -2.84, 3.36, -0.08, -5.12, 2.36), # bedroom
                create_room(0.86, -6.93, 2.93, 3.0, -7.85, 2.39), # bedroom
                create_room(2.53, -3.9, 3.15, 3.19, -2.98, 2.81), #bathroom
                create_room(0.95, -3.20, 3.09, 1.76, -4.96, 2.10), #corridor
                create_room(3.92, -4.79, 2.23, 5.07, -4.90, 1.30), #staircase
                create_room(3.7, -8.03, 0.69, 3.5, -6.28, -0.04), #entrance hall
                create_room(1.44, -8.22, 0.37, -1.75, -2.88, -0.27), #livingroom
                create_room(0.12, -0.65, 0.48, 0.89, 1.43, -0.39), #livingroom
                create_room(2.21, 1.98, 0.00, 3.43, 3.63, -0.31), #small table room
                create_room(4.16, 0.07, 0.09, 2.96, -0.24, -0.17), #bathroom
                create_room(4.25, -1.24, 0.52, 2.32, -3.73, -0.46) # workroom
            ],
            "apartment_1": [
                create_room(-1.20, 0.42, 0.53, 0.40, 5.59, -0.22), # livingroom
                create_room(2.76, 5.69, 0.49, 6.5, 4.1, -0.22) # meetingroom
            ],
            "apartment_2": [
                create_room(-1.25, 1.03, 0.79, -0.09, -0.94, -0.26), # workroom
                create_room(1.90, -0.76, 0.35, 5.84, 0.67, -0.50), # bedroom
                create_room(5.69, 3.0, 0.38, 3.97, 4.08, -0.32), # livingroom
                create_room(3.11, 5.86, 0.38, 5.35, 7.75, 0.04) # eatingroom
            ], 
            "frl_apartment_0": [
                create_room(0.55, -4.08, 0.31, 4.58, -1.98, -0.24),
                create_room(4.89, -5.53, -0.31, 3.24, -7.87, 0.36),
                create_room(0.019, 1.89, 0.77, 0.51, 0.45, -0.29)
            ],
            "frl_apartment_1": [
                create_room(-1.96, 0.09, -0.66, -0.79, 0.65, 0.46),
                create_room(1.27, 0.56, 0.78, 3.74, 4.14, 0.04),
                create_room(5.45, 4.85, 0.06, 7.45, 3.51, 0.53)
            ],
            "hotel_0": [
                create_room(-1.56, -0.26, 1.08, 1.4, 0.95, 0.113),
                create_room(2.64, 1.4, 1.0, 4.51, 0.98, -0.07),
                create_room(4.71, -0.15, 0.90, 3.69, -0.65, 0.34)
            ],
            "office_0": [
                create_room(-0.98, 0.36, 1.21, 0.85, -1.56, 0.27)
            ],
            "office_1": [
                create_room(-0.12, -0.50, 1.37, 0.64, 0.93, 0.30)
            ],
            "room_0": [
                create_room(0.0, 0.00, 0.51, 4.9, 2.03, 0.02)
            ],
            "room_1": [
                create_room(0.26, 0.22, 0.98, -3.82, -0.19, -0.13)
            ],
            "room_2": [
                create_room(0.75, -1.13, 0.02, 4.22, -0.33, -1.15)
            ]
        }
        
    @staticmethod
    def filename_from_frame_number(frame_number):
        return f"{frame_number:05d}.png"
    
    def load_scene_semantic_dict(self, scene):
        with open(os.path.join(self._dataset_path, scene, 'habitat', 'info_semantic.json'), 'r') as f:
            return json.load(f)
        
    def fix_semantic_observation(self, semantic_observation, scene_dict):
        # The labels of images collected by Habitat are instance ids
        # transfer instance to semantic
        instance_id_to_semantic_label_id = np.array(scene_dict["id_to_label"])
        semantic_img = instance_id_to_semantic_label_id[semantic_observation]               
                
        return semantic_img

    def save_color_observation(self, observation, frame_number, out_folder, scene):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        color_observation = observation["color_sensor"]
        color_img = Image.fromarray(color_observation)
        # save color images
        color_img.save(os.path.join(out_folder, scene+self.filename_from_frame_number(frame_number)))
        self._last_frame = np.array(color_img)

    def save_semantic_observation(self, observation, frame_number, out_folder, scene_dict, scene):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)      
        semantic = self.fix_semantic_observation(observation["semantic_sensor"], scene_dict)
        semantic_img = Image.new("L", (semantic.shape[1], semantic.shape[0]))        
        semantic_img.putdata(semantic.flatten())
        # save semantic images
        semantic_img.save(os.path.join(out_folder, scene+self.filename_from_frame_number(frame_number)))       
        self._last_semantic_frame = np.array(semantic_img)

    def save_depth_observation(self, observation, frame_number, out_folder,scene):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        depth_observation = observation["depth_sensor"]
        depth_img = Image.fromarray(
            (depth_observation / 10 * 255).astype(np.uint8), mode="L")
        # save depth images
        depth_img.save(os.path.join(out_folder, scene+self.filename_from_frame_number(frame_number)))
        self._last_depth_frame = np.array(depth_img)

    def save_observations(self, observation, frame_number, out_folder, split_name, scene_dict, scene):
        self.save_color_observation(observation, frame_number, os.path.join(out_folder, 'images', split_name), scene)
        self.save_semantic_observation(observation, frame_number, os.path.join(out_folder, 'annotations', f"{split_name}"), scene_dict, scene)
        self.save_depth_observation(observation, frame_number, os.path.join(out_folder, 'depth', split_name), scene)
        
            
    def generate(self, out_folder, split_name, frames_per_room=100):
        
        settings = {}
        print(out_folder)
        settings['width'] = self._width
        settings['height'] = self._height
        settings["sensor_height"] = 0
        settings["color_sensor"] = True
        settings["depth_sensor"] = True
        settings["semantic_sensor"] = True
        settings["silent"] = True
        
        current_frame = 0
        total_frames = 0   
        
        for scene in self._scenes:
            # setup the simulator
            settings["scene"] = os.path.join(self._dataset_path, scene, "habitat", "mesh_semantic.ply")
            cfg = make_cfg(settings)
            simulator = habitat_sim.Simulator(cfg)
            
            # load semantic information for scene
            scene_semantic_dict = self.load_scene_semantic_dict(scene)
            
            # generate data for each room
            
            gt_poses = np.load(f'../hw1/data_collection/{out_folder}/GT_pose.npy')
            total_frames = len(gt_poses)
            for pose in gt_poses:
                agent = simulator.get_agent(0)
                agent_state = agent.get_state()
                random_state = AgentState()
                # Translation is a 3 dimensional vector
                random_state.position[0] = pose[0]
                random_state.position[1] = pose[1]
                random_state.position[2] = pose[2]
                # Rotation is a quaternion
                random_state.rotation.w = pose[3]
                random_state.rotation.x = pose[4]
                random_state.rotation.y = pose[5]
                random_state.rotation.z = pose[6]
                
                list(agent_state.position)+list(agent_state.rotation.components)
                agent_state.sensor_states = {}
                agent.set_state(random_state)
                
                # do the actual rendering
                observations = simulator.get_sensor_observations()                   
                self.save_observations(observations, current_frame, out_folder, 
                                       split_name, scene_semantic_dict, scene)                    
                
                print(f'Saved image {current_frame+1}/{total_frames}')
                current_frame += 1
                    
            simulator.close()
            
            del simulator
            

def main():
    """Main function of the program.
    """
    # python data_generator.py --dataset [dataset folder] --output [output folder] 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Folder containing Replica dataset")
    parser.add_argument("--output", type=str, choices=['first_floor', 'second_floor'], help="Output folder")
    args = parser.parse_args()
    
    # adjust [frames_per_room] to collect arbitrary number of  images
    generator = Generator(path=args.dataset)
    # generator.generate(out_folder=args.output, 
    #                    split_name='train',
    #                    frames_per_room=1)
    generator.generate(out_folder=args.output, 
                       split_name='val',
                       frames_per_room=1)
    # generator.generate(out_folder=args.output, 
    #                    split_name='test',
    #                    frames_per_room=1)
    
if __name__ == "__main__":
    main()
