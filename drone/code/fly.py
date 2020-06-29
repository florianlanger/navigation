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
from djitellopy import Tello
import cv2, math, time
import shutil
#from pythonzenity import Entry

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/Cambridge Academic/MPhil project/navigation/drone"))

from models.action_predictor import Decoder_Basic
from visualisations import plot_trajectory, plot_action_predictions
from models.pose import Pose_Model



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

def during_flight(tello,pose_model,action_predictor,exp_path):
    counter = 0

    keep_going = True
    mode = 'manual'
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



def initialise_history_dict():
    history_dict = {}
    history_dict['statistics'] = {}
    history_dict['predicted_poses'] = []
    history_dict['predicted_targets'] = []
    history_dict['action_predictions'] = []
    history_dict['image_names'] = []
    history_dict['counter'] = 0
    history_dict['target_counter'] = 0
    history_dict['terminated'] = False
    return history_dict

def main():

    # Bookkeeping 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = '{}/../experiments/{}'.format(dir_path, datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
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