

import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models import resnet18
from datetime import datetime
import cv2
import sys
import json
import pickle
import ast
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from visualisations import plot_trajectory

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation"))
from graph_network.code.graphs.conversions import Converter

from pose_room_books.code.models.model import Pose_Model


def load_image(sequences_folder,name,test_folder):
    image = Image.open(sequences_folder +'/images/' + name)
    image = image.resize((128,96))
    image.save(test_folder+'/images/'+name)
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0)
    return image

def load_converter(corners_no_fly_zone):
    min_pos = torch.tensor([0.0,0.0,0.0,0.]) #.cuda()
    steps = torch.tensor([0.2,0.2,0.2,0.25]) #.cuda()
    max_pos = torch.tensor([4.8,4.8,1.8,0.75]) #.cuda()
    number_poses = 20736
    converter = Converter(min_pos,max_pos,steps,number_poses,corners_no_fly_zone)
    return converter

def global_transformation(predicted_pose):
    position = predicted_pose[:3]*1000
    angle = np.array([(predicted_pose[3]-0.37)%1 * 360])

    rotation = R.from_euler('z',320,degrees=True)
    new_position = 0.00217*rotation.apply(position) + [2.9,2.05,1.05]
    return np.concatenate((new_position,angle))

def perform_test(pose_model,config,test_folder,dir_path,objects,converter,image_names,image_folder):
    
    history_dict = {}
    history_dict['statistics'] = {}
    history_dict['true_poses'] = np.zeros((config["max_moves"],4))
    history_dict['predicted_poses'] = np.zeros((config["max_moves"],4))
    history_dict['predicted_targets'] = np.zeros((config["max_moves"],4))
    history_dict['action_predictions'] = np.zeros((config["max_moves"],10))
    history_dict['image_names'] = image_names
    history_dict['counter'] = 0
    history_dict['last_counter'] = 0
    history_dict['target_counter'] = 0
    history_dict['terminated'] = False
    history_dict['global_probabilities'] = np.zeros(config["probability_grid"]["points_per_dim"])

    # Repeat until network says terminate or until have reached max_moves
    for i in tqdm(range(len(image_names))):

        predicted_pose = pose_model(load_image(image_folder,image_names[i],test_folder))[0]

        predicted_pose = global_transformation(predicted_pose)

        history_dict['predicted_poses'][history_dict['counter']] = predicted_pose.round(4)
            
        #create visualisations
        plot_trajectory(history_dict, test_folder,converter)
        history_dict['counter'] +=1

    return history_dict

def objects_to_no_fly(objects):
    objects_no_fly = torch.zeros((len(objects),2,3))
    for i,key in enumerate(objects):
        objects_no_fly[i,0] = torch.from_numpy(np.array(objects[key]['dimensions'][:3]) - objects[key]['dimensions'][3:6])
        objects_no_fly[i,1] = torch.from_numpy(np.array(objects[key]['dimensions'][:3]) + objects[key]['dimensions'][3:6])
    return objects_no_fly

def load_model(dir_path,config):

    pretrained_model = resnet18(pretrained=True)
    pretrained_model.fc = nn.Sequential()
    pose_model = Pose_Model(pretrained_model,512)

    #pose_model.load_state_dict(torch.load(dir_path + '/../drone/networks/pose_networks/exp_02_new_room_save_time_17_17_25_date_08_08_2020_epoch_13_model.pth',map_location=torch.device('cpu')))
    pose_model.load_state_dict(torch.load(dir_path + '/../drone/networks/pose_networks/exp_02_new_room_time_16_48_39_date_08_08_2020_epoch_101_model.pth',map_location=torch.device('cpu')))
    pose_model.eval()

    return pose_model

def main():
   

        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = json.load(open(dir_path + '/config.json', 'r'))
        
        sequence_folder = '../drone/experiments/fly_sequences/seq_2_time_19_07_35_date_08_08_2020'
        #sequence_folder = '../drone/experiments/fly_sequences/seq_1_time_19_10_36_date_08_08_2020'

        with open(sequence_folder + '/image_names.txt') as file:
            line = file.readline()
            image_names = ast.literal_eval(line)

        objects = config[config["room"]]["objects"]
        objects_no_fly = objects_to_no_fly(objects)
        converter = load_converter(objects_no_fly)
        print(converter.corners_no_fly_zone)

        dt_string = datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y")
        test_folder =  "{}/tests/{}_{}".format(dir_path,config["name"],dt_string)
        os.mkdir(test_folder)
        os.mkdir(test_folder + '/trajectories')
        os.mkdir(test_folder + '/images')

        #load model
        pose_model = load_model(dir_path,config)

        # need list of images and images
        with torch.no_grad():
            for j in range(config["number_tests"]):
                history_dict = perform_test(pose_model,config,test_folder,dir_path,objects,converter,image_names,sequence_folder)

                dict_file = test_folder + '/history_dict'
                f = open(dict_file + '.txt','a')
                f.write(str(history_dict))
                f.close()
                np.save(dict_file+'.npy', history_dict)

if __name__ == "__main__":
    main()



