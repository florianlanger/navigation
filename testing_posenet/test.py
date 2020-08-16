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


def load_image(folder_path,name):
    image = Image.open(folder_path +'/images/' + name)
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0)#.cuda()
    return image


def perform_test(pose_model,config,test_folder,dir_path,objects,image_names):
    
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
    for i in range(len(image_names)):

        predicted_pose = pose_model(load_image(test_folder,current_name))[0]

        predicted_pose[3] = predicted_pose[3] % 1.
        history_dict['true_poses'][history_dict['counter']] = true_pose.cpu().numpy().round(4)
        history_dict['predicted_poses'][history_dict['counter']] = predicted_pose.cpu().numpy().round(4)
            
        print('Predicted Pose: {} Predicted Target: {}'.format(list(predicted_pose.cpu().numpy().round(3)),list(predicted_target_pose.cpu().numpy().round(3))))


        #create visualisations
        plot_trajectory(history_dict, test_folder,converter)
        visualise_poses(history_dict,test_folder)
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
    pose_model.load_state_dict(torch.load(dir_path + '/../drone/networks/pose.pth',map_location=torch.device('cpu')))
    #pose_model.load_state_dict(torch.load(dir_path + '/../pose/experiments/' + config["pose_model"]))
    pose_model.eval()

    return pose_model

def main():
   

        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = json.load(open(dir_path + '/config.json', 'r'))

        objects = config[config["room"]]["objects"]
        objects_no_fly = objects_to_no_fly(objects)

        #load model
        pose_model = load_model(dir_path,config)

        # need list of images and images


        with torch.no_grad():
            for j in range(config["number_tests"]):


                history_dict = perform_test(pose_model,config,test_folder,dir_path,objects,image_names)

                dict_file = test_folder + '/history_dict'
                f = open(dict_file + '.txt','a')
                f.write(str(history_dict))
                f.close()
                np.save(dict_file+'.npy', history_dict)

if __name__ == "__main__":
    main()



