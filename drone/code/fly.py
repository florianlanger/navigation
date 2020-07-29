#Press ESC for drone to land.
# Change mode between manual (m) where you control the drone with keyboard
# and (n) automatic where drone flies by itself. 
# In Manual mode Forward (w), Backward (s), Left (a), Right(d), Up (r), Down (f), Rotate ACW(q), Rotate CW (e)  



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
import subprocess
import cv2, math, time
import shutil
import time
import networkx as nx

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation/drone"))

from tellopy.modified_tellopy import Tello

from models.action_predictor import Decoder_Basic
from visualisations import plot_trajectory, plot_action_predictions
from models.pose import Pose_Model

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation"))

from graph_network.code.graphs.no_rotation import index_to_position, position_to_index

from testing.test import update_target_probabilities_from_speech

from train_target_predictor.code.model import LSTM_model
from train_target_predictor.code.data import Target_Predictor_Dataset


#make experiment
def load_models(dir_path):

    pretrained_model = resnet18()
    pretrained_model.fc = nn.Sequential()
    pose_model = Pose_Model(pretrained_model,512)
    pose_model.load_state_dict(torch.load(dir_path + '/../networks/pose.pth',map_location=torch.device('cpu')))
    pose_model.eval()

    action_predictor = Decoder_Basic(10)
    action_predictor.load_state_dict(torch.load(dir_path + '/../networks/exp_6_small_angles_hard_pairs_time_15_02_48_date_02_06_2020_last_epoch_model.pth',map_location=torch.device('cpu')))
    action_predictor.eval()

    return pose_model,action_predictor


def make_directories(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path + '/images')
    os.mkdir(exp_path + '/trajectories')
    os.mkdir(exp_path + '/action_predictions')
    os.mkdir(exp_path + '/recordings')
    shutil.copytree(exp_path +'/../../code',exp_path +'/code')

def find_pose(pose_model,image_name,exp_path):
    image = Image.open(exp_path +'/images/' + image_name)
    image = image.crop((120,0,720,720))
    image = image.resize((100,100))
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0)
    pose = pose_model(image)[0]
    print(pose)
    return pose

def instruction_to_index(text_instruction):
    dict_instruction_to_index = {}
    dict_instruction_to_index['Stay'] = 0
    dict_instruction_to_index['Go forward'] = 1
    dict_instruction_to_index['Go backward'] = 2
    dict_instruction_to_index['Go left'] = 3
    dict_instruction_to_index['Go right'] = 4
    dict_instruction_to_index['Go up'] = 5
    dict_instruction_to_index['Go down'] = 6 
    dict_instruction_to_index['Rotate acw'] = 7
    dict_instruction_to_index['Rotate cw'] = 8
    dict_instruction_to_index['Keep going'] = 9
    return dict_instruction_to_index[text_instruction]

def execute_action(tello,action_index):
    if action_index == 0:
        pass
    elif action_index == 1 or action_index == 2 or action_index == 3 or action_index == 4:
        tello.move_forward(10)
    elif action_index == 5:
        tello.move_up(10)
    elif action_index == 6:
        tello.move_down(10)
    elif action_index == 7:
        tello.rotate_counter_clockwise(30)
    elif action_index == 8:
        tello.rotate_clockwise(30)
    elif action_index == 9:
        print('-----------------------------------')
        print('Terminate')

def action_index_to_text(action_index):
    if action_index == 0:
        return 'Stay'
    elif action_index == 1 or action_index == 2 or action_index == 3 or action_index == 4:
        return 'Go Forward'
    elif action_index == 5:
        return 'Go up'
    elif action_index == 6:
        return 'Go down'
    elif action_index == 7:
        return 'Rotate ACW'
    elif action_index == 8:
        return 'Rotate CW'
    elif action_index == 9:
        return 'Terminate'

def check_user_input(tello,mode):
    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        tello.land()
        cv2.destroyAllWindows()
        return False,'end_mode'
    else:
        if key == ord('w'):
            tello.move_forward(30)
        elif key == ord('s'):
            tello.move_back(30)
        elif key == ord('a'):
            tello.move_left(30)
        elif key == ord('d'):
            tello.move_right(30)
        elif key == ord('e'):
            tello.rotate_clockwise(30)
        elif key == ord('q'):
            tello.rotate_counter_clockwise(30)
        elif key == ord('r'):
            tello.move_up(30)
        elif key == ord('f'):
            tello.move_down(30)
        elif key == ord('n'):
            mode = 'automatic'
        elif key == ord('m'):
            mode = 'manual'
        return True,mode

def execute_command(tello,command):
    if command == "pos x":
        tello.move_forward(20)
    elif command == "neg x":
        tello.move_back(20)
    elif command == "pos y":
        tello.move_left(20)
    elif command == "neg y":
        tello.move_right(20)
    elif command == "pos z":
        tello.move_up(20)
    elif command == "neg z":
        tello.move_down(20)

def during_flight(tello,pose_model,action_predictor,exp_path):
    counter = 0

    keep_going = True
    scenario = 'scenario_01.txt'
    mode = 'automatic'
    step_number = 0
    while keep_going == True:
        while mode == 'automatic':
            if counter == 0:
                history_dict = initialise_history_dict()
                #entry = Entry(text="Where should I fly to?", entry_text="Fly over the table")
                #print(entry)
                # find target from text here
                target_pose = torch.zeros((4,))
                target_pose[2] += 1.
                history_dict['predicted_targets'].append(target_pose.numpy())

            if(counter%1000 ==0):
                automatic_flight(tello,pose_model,action_predictor,history_dict,target_pose,exp_path,counter)
                history_dict['counter'] +=1            

            keep_going,mode = check_user_input(tello,mode)

            counter=counter+1
        

        while mode == 'follow_scenario':
            if(counter%1000 ==0):
                follow_commands(tello,exp_path,counter)
                step_number += 1           

            counter=counter+1

        while mode == 'manual':
            print('in mode manual')
            img = tello.get_frame_read().frame
            keep_going,mode = check_user_input(tello,mode)
            cv2.imshow("drone", img)
    
    dict_file = exp_path + '/history_dict'
    f = open(dict_file + '.txt','a')
    f.write(str(history_dict))
    f.close()
    np.save(dict_file+'.npy', history_dict)


def automatic_flight(tello,pose_model,action_predictor,history_dict,target_pose,exp_path,counter):
    
    #read current frame and save image
    img = tello.get_frame_read().frame
    image_name = 'image_{}.png'.format(str(counter).zfill(6))
    history_dict['image_names'].append(image_name)
    cv2.imwrite(exp_path + '/images/' + image_name,img)

    # Find predicted pose
    predicted_pose = find_pose(pose_model,image_name,exp_path)
    predicted_pose[3] = predicted_pose[3] % 1.
    history_dict['predicted_poses'].append(predicted_pose.numpy().round(4))

    #Calculate action predictions
    action_predictions = action_predictor(torch.cat((predicted_pose,target_pose),dim=0).unsqueeze(0))[0]
    _,action_index = torch.max(action_predictions,dim=0)
    history_dict['action_predictions'].append(action_predictions.numpy().round(4))

    # Execute actions
    execute_action(tello,action_index)

    #Display live image
    cv2.putText(img, 'Predicted Pose: {}'.format(list(predicted_pose.numpy().round(3))), (5, 720 - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, 'Predicted Target: {}'.format(list(target_pose.numpy())), (5, 720 - 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, 'Predicted Action: {}'.format(action_index_to_text(action_index)), (5, 720 - 75),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("drone", img)

    # Create visualisations
    plot_trajectory(history_dict, exp_path + '/trajectories')
    plot_action_predictions(history_dict,exp_path + '/action_predictions')


def initialise_history_dict(config):
    history_dict = {}
    history_dict['statistics'] = {}
    history_dict['predicted_poses'] = []
    history_dict['predicted_targets'] = []
    history_dict['action_predictions'] = []
    history_dict['image_names'] = []
    history_dict['counter'] = 0
    history_dict['target_counter'] = 0
    history_dict['terminated'] = False
    history_dict["probability_contributions"] = []
    history_dict['global_probabilities'] = np.zeros(config["probability_grid"]["points_per_dim"])
    return history_dict


def find_list_of_commands(start_position,target_position):
    scene = torch.tensor([3.8,3.2,1.6,0.2,0.2,0.4])
    G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graph_network/graphs/own_room_no_rotation.gpickle')
    start_index = position_to_index(start_position,scene)
    target_index = position_to_index(target_position,scene)
    shortest_path = nx.shortest_path(G,start_index,target_index)
    
    list_commands = []
    for i in range(len(shortest_path)-1):
        list_commands.append(G[shortest_path[i]][shortest_path[i+1]][0]['action'])
    print(list_commands)
    return list_commands

def convert_objects(objects):
        new_objects = {}
        for key in objects:
            old_dims = objects[key].numpy()
            new_dimensions = np.zeros(6)
            new_dimensions[:3] = old_dims[3:6] + 0.5 * old_dims[:3]
            new_dimensions[3:6] =  0.5 * old_dims[:3]
            new_objects[key] = new_dimensions
        return new_objects

def main():

    # Bookkeeping 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = '{}/../experiments/{}'.format(dir_path, datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    make_directories(exp_path)

    config = json.load(open(dir_path + '/config.json', 'r'))


    tello = Tello()
    tello.connect()
    tello.streamon()

    tello.takeoff()

    tello.move_forward(30)
    time.sleep(0.3)
    tello.move_forward(30)
    tello.land()
    tello.end()
    """

    counter = 0
    print('go')
    while True:
        img = tello.get_frame_read().frame
        image_name = 'image_{}.png'.format(str(counter).zfill(3))
        cv2.imwrite(exp_path + '/images/' + image_name,img)
        time.sleep(0.5)
        counter += 1
        print(counter)


    #Define these
    start_position = torch.tensor([1.4,2.0,0.8])
    #target_position = torch.tensor([2.4,0.4,0.4])
    history_dict = initialise_history_dict(config)
    
    # Load dataset and lstm model
    text_dataset = Target_Predictor_Dataset(dir_path + '/../../target_pose/training_data1/data_transcribed_new.csv',250)
    lstm_model = LSTM_model(5,32,63)
    lstm_model.load_state_dict(torch.load(dir_path + '/../networks/debug_testing_time_14_44_52_date_15_07_2020_epoch_241_model.pth',map_location=torch.device('cpu')))
    lstm_model.eval()


    # define objects 
    objects = {
                'big cupboard': torch.tensor([0.63,1.23,2.65,0,2.42,0]),
                'sideboard': torch.tensor([1.2,0.6,0.75,0,0,0]),
                'table': torch.tensor([1.8,0.9,0.2,1.2,0,0.65]),
                'couch': torch.tensor([1.56,0.88,0.7,2.38,2.78,0]),
                'stool': torch.tensor([0.6,0.4,0.4,2.38,2.18,0]),
                'small cupboard': torch.tensor([0.34,0.44,0.7,3.74,2.2,0.]),
                'printer': torch.tensor([0.5,0.5,0.4,3.4,1.25,0]),
                'lamp': torch.tensor([0.25,0.85,0.7,2.28,0,0.75])
            }

    objects_center_format = convert_objects(objects)

    # Find target position
    predicted_target_pose,contribution_global_probabilities = update_target_probabilities_from_speech(exp_path,0,lstm_model,text_dataset,objects_center_format,config)
    print('target_pose',predicted_target_pose)

    history_dict["probability_contributions"].append(contribution_global_probabilities)
    history_dict["global_probabilities"] += contribution_global_probabilities

    target_position = predicted_target_pose[:3]
    list_commands = find_list_of_commands(start_position,target_position)

    #Load models
    #pose_model,action_predictor = load_models(dir_path)

    # Initialise drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    # #Take off
    tello.takeoff()

    tello.move_forward(30)
    time.sleep(0.3)
    tello.move_forward(30)
    tello.land()
    tello.end()

    for command in list_commands:
        print(command)
        execute_command(tello,command)
        time.sleep(0.3)


    tello.rotate_counter_clockwise(30)
    tello.rotate_clockwise(30)

    tello.land()
    tello.end()


  
            


    # #Enter flight mode
    # with torch.no_grad():
    #     during_flight(tello,pose_model,action_predictor,exp_path)
    #     tello.end()
    """


if __name__ == "__main__":
    with torch.no_grad():
        main()