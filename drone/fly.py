# simple example demonstrating how to control a Tello using your keyboard.
# For a more fully featured example see manual-control-pygame.py
# 
# Use W, A, S, D for moving, E, Q for rotating and R, F for going up and down.
# When starting the script the Tello will takeoff, pressing ESC makes it land
#  and the script exit.

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
from djitellopy import Tello
import cv2, math, time

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/Cambridge Academic/MPhil project/navigation/drone"))
sys.path.append(os.path.abspath("/Users/legend98/Google Drive/Cambridge Academic/MPhil project/navigation"))

from action_predictor.code.models.decoder import Decoder_Basic
from pose.code.models.model import Pose_Model



#make experiment
def load_models(dir_path):

    pretrained_model = resnet18()
    pretrained_model.fc = nn.Sequential()
    pose_model = Pose_Model(pretrained_model,512)
    pose_model.load_state_dict(torch.load(dir_path + '/networks/pose.pth',map_location=torch.device('cpu')))
    pose_model.eval()

    action_predictor = Decoder_Basic(10)
    action_predictor.load_state_dict(torch.load(dir_path + '/networks/exp_6_small_angles_hard_pairs_time_15_02_48_date_02_06_2020_last_epoch_model.pth',map_location=torch.device('cpu')))
    action_predictor.eval()

    return pose_model,action_predictor


def make_directories(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path + '/images')

def find_pose(pose_model,image_name,exp_path):
    image = Image.open(exp_path +'/images/' + image_name)
    image = image.crop((120,0,720,720))
    image = image.resize((100,100))
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0)
    pose = pose_model(image)[0]
    print(pose)
    return pose

def find_next_action(action_predictor,predicted_pose,target_pose):
    concat_poses = torch.cat((predicted_pose,target_pose),dim=0).unsqueeze(0)
    action_predictions = action_predictor(concat_poses)[0]
    print('Action predictions',action_predictions)
    _,action_index = torch.max(action_predictions,dim=0)
    return action_index

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
        tello.move_forward(30)
    elif action_index == 5:
        tello.move_up(30)
    elif action_index == 6:
        tello.move_down(30)
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
    print('key',key)
    if key == 27: # ESC
        tello.land()
        cv2.destroyAllWindows()
        return False,'end_mode'
    else:
        print('ord(w)',ord('w'))
        if key == ord('w'):
            print('press w')
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

def during_flight(tello,pose_model,action_predictor,exp_path):
    frame_read = tello.get_frame_read()
    counter = 0
    target_pose = torch.zeros((4,))
    target_pose[2] += 1.
    
    keep_going = True
    mode = 'manual'
    while keep_going == True:
        while mode == 'automatic':
            if(counter%1000 ==0):
                img = frame_read.frame
                image_name = 'image_{}.png'.format(str(counter).zfill(6))
                cv2.imwrite(exp_path + '/images/' + image_name,img)
                predicted_pose = find_pose(pose_model,image_name,exp_path)
                predicted_pose[3] = predicted_pose[3] % 1.
                action_index = find_next_action(action_predictor,predicted_pose,target_pose)
                execute_action(tello,action_index)
                cv2.putText(img, 'Predicted Pose: {}'.format(list(predicted_pose.numpy().round(3))), (5, 720 - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, 'Predicted Target: {}'.format(list(target_pose.numpy())), (5, 720 - 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, 'Predicted Action: {}'.format(action_index_to_text(action_index)), (5, 720 - 75),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("drone", img)
            

            keep_going,mode = check_user_input(tello,mode)

            counter=counter+1
        
        while mode == 'manual':
            print('in mode manual')
            img = frame_read.frame
            keep_going,mode = check_user_input(tello,mode)
            cv2.imshow("drone", img)


def main():

    # Bookkeeping 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = '{}/experiments/{}'.format(dir_path, datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    make_directories(exp_path)

    #Load models
    pose_model,action_predictor = load_models(dir_path)

    # Initialise drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    #Take off
    tello.takeoff()

    #Enter flight mode
    with torch.no_grad():
        during_flight(tello,pose_model,action_predictor,exp_path)
        tello.end()


if __name__ == "__main__":
    main()