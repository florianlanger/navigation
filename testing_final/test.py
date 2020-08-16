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
import fasttext
import time
import networkx as nx
from scipy.spatial.transform import Rotation as R

import sounddevice as sd
from scipy.io.wavfile import write
import wavio
import azure.cognitiveservices.speech as speechsdk

from visualisations import plot_trajectory,plot_global_probability, plot_combined

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation"))
from graph_network.code.graphs.conversions import Converter
from action_predictor.code.models.decoder import Decoder_Basic
from pose.code.models.model import Pose_Model
from target_adjuster.code.models.model import Target_Adjuster
from target_pose.target_pose import find_point,filter_text
from train_target_predictor.code.data import Target_Predictor_Dataset
from train_target_predictor.code.model import LSTM_model
from train_target_predictor_mdn.code.model import MixtureDensityNetwork
from drone.code.tellopy.modified_tellopy import Tello

def global_transformation(predicted_pose):
    position = predicted_pose[:3]*1000
    angle = np.array([(predicted_pose[3]-0.37)%1 * 360])
    rotation = R.from_euler('z',320,degrees=True)
    new_position = 0.00217*rotation.apply(position) + [2.9,2.05,1.05]
    return np.concatenate((new_position,angle))


def load_image(folder_path,name):
    image = Image.open(folder_path +'/images/' + name)
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0)
    return image

def execute_action(tello,action):
    if action == "pos x" or action == "neg x" or action == "pos y" or action == "neg y":
        tello.move_forward(20)
    elif action == "pos z":
        tello.move_up(20)
    elif action == "neg z":
        tello.move_down(20)
    elif action == "rot +":
        tello.rotate_counter_clockwise(90)
    elif action == "rot -":
        tello.rotate_clockwise(90)

def initialise_history_dict(config):
    history_dict = {}
    history_dict['statistics'] = {}
    history_dict['predicted_poses'] = np.zeros((config["max_moves"],4))
    history_dict['predicted_targets'] = np.zeros((config["max_moves"],4))
    history_dict['image_names'] = []
    history_dict['counter'] = 0
    history_dict['last_counter'] = 0
    history_dict['target_counter'] = 0
    history_dict['terminated'] = False
    history_dict['actions'] = []
    history_dict['global_probabilities'] = np.zeros(config["probability_grid"]["points_per_dim"])
    return history_dict

def start_tello():
    tello = Tello()
    tello.connect()
    tello.streamon()
    tello.takeoff()
    return tello

def save_image(image_name,tello):
    img = tello.get_frame_read().frame
    img = cv2.resize(img,(128,96))
    cv2.imwrite(image_name,img)

def find_action(start_pose,target_pose,G,converter):
    start_pose = torch.from_numpy(start_pose)
    start_pose[2] = start_pose[2] + 0.3
    start_pose[3] = start_pose[3]/360.
    print('before graph',start_pose)
    index_start = converter.pose_to_index(start_pose)
    index_end = converter.pose_to_index(target_pose)
    shortest_path = nx.shortest_path(G,index_start,index_end)
    actions = []
    for i in range(7):
        actions.append(G[shortest_path[i]][shortest_path[i+1]][0]['action'])
    return actions

def perform_test(pose_model,mdn_model,config,converter,test_folder,dir_path,objects,G):
    history_dict = initialise_history_dict(config)

    tello = start_tello()

    start_time = int(round(time.time() * 1000))

    while history_dict['terminated'] == False:
        #Get current camera image
        current_time = int(round(time.time() * 1000))
        image_name = 'image_{}.png'.format(str(current_time - start_time).zfill(6))
        history_dict["image_names"].append(image_name)
        save_image(test_folder+ '/images/'+image_name,tello)
        
        # Predict current pose
        predicted_pose = pose_model(load_image(test_folder,image_name))[0]

        # Transform pose
        predicted_pose = global_transformation(predicted_pose)
        history_dict['predicted_poses'][history_dict['counter']] = predicted_pose.round(4)

       # if key press
        key = cv2.waitKey(1) & 0xff
        if key == 27: # ESC
            tello.land()
            tello.end()
            cv2.destroyAllWindows()
            history_dict['terminated'] == True

        elif history_dict['counter'] == 0 or key == ord('r'):
            contribution_global_probabilities,text,anchor_object = update_target_probabilities_from_speech(test_folder,history_dict['target_counter'],mdn_model,objects,config,history_dict,tello)
            history_dict['global_probabilities'] = history_dict['global_probabilities'] + contribution_global_probabilities
            predicted_target_pose = torch.tensor([3.,2.,1.4,0])#find_current_target(history_dict["global_probabilities"],config)
            history_dict['predicted_targets'][history_dict['target_counter']] = predicted_target_pose.cpu().numpy().round(4)
            history_dict['last_counter'] = history_dict['counter']
            plot_global_probability(history_dict['global_probabilities'],test_folder,converter,config,history_dict['target_counter'],'total')
            plot_global_probability(contribution_global_probabilities,test_folder,converter,config,history_dict['target_counter'],'last',text,anchor_object=anchor_object)
            history_dict['target_counter'] += 1
            
        print('Predicted Pose: {} Predicted Target: {}'.format(list(predicted_pose.round(3)),list(predicted_target_pose.numpy().round(3))))

        print(predicted_pose,)
        actions = find_action(predicted_pose,predicted_target_pose,G,converter)
        for action in actions:
            history_dict['actions'].append(action)
            execute_action(tello,action)

        #create visualisations
        plot_trajectory(history_dict, test_folder,converter)
        plot_combined(history_dict,test_folder)
        
        img = cv2.imread('{}/combined/combined_{}.png'.format(test_folder,str(history_dict["counter"]).zfill(2)))
        cv2.imshow("drone", img)

        history_dict['counter'] +=1

    return history_dict

def voice_command(path,counter):
    input('Press anything to start recording')
    print('Start recording ...')
    duration = 5  # seconds
    fs = 44100
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print('Finished recording')
    audio_path = path+'/recordings/recording_{}.npy'.format(str(counter).zfill(4))
    np.save(audio_path,myrecording)
    return audio_path

def update_target_probabilities_from_speech(test_folder,counter,mdn_model,objects,config,history_dict,tello):
    #tello.rotate_counter_clockwise(1)
    #audio_file = voice_command(test_folder,counter)
    tello.rotate_clockwise(1)
    #original_text = transcribe_audio(audio_file)

    original_text = "to the right of the bed on central height"
    text,anchor_object = preprocess_text(original_text,objects,history_dict,tello)
    print(text)

    pi,normal = mdn_model([text])

    probability_contribution = find_probability_contribution_from_parameters(pi,normal,objects[anchor_object],config)
    return probability_contribution,original_text,anchor_object

def find_probability_contribution_from_parameters(pi,normal,anchor_object,config):

    points_per_dim = config["probability_grid"]["points_per_dim"]
    global_probabilities = np.zeros((points_per_dim[0],points_per_dim[1],points_per_dim[2]))
    for i in range(points_per_dim[0]):
        for j in range(points_per_dim[1]):
            for k in range(points_per_dim[2]):
               
                indices = np.array([i,j,k])
                position = torch.from_numpy(config["probability_grid"]["min_position"] + indices * config["probability_grid"]["step_sizes"])

                #print('actual position', position)
                position =  (position - torch.tensor(anchor_object["dimensions"][:3]) ) /(torch.tensor(anchor_object["scaling"])*2*torch.tensor(anchor_object["dimensions"][3:6]))
                position = position.unsqueeze(0).unsqueeze(0)

                # print(position.shape)
                probabilities = torch.exp(torch.sum(normal.log_prob(position),dim=2))
                
                probability = torch.sum(pi.probs * probabilities,dim=1).item()
                
                global_probabilities[i,j,k] = probability


    global_probabilities = global_probabilities / np.max(global_probabilities)
    return global_probabilities

def find_current_target(global_probabilities,config):
        points_per_dim = config["probability_grid"]["points_per_dim"]
        index = np.argmax(global_probabilities.flatten())
        index = index.item()
        indices = np.array([index // (points_per_dim[1]*points_per_dim[2]),  (index % (points_per_dim[1]*points_per_dim[2]))  // points_per_dim[2], index % points_per_dim[2]])
        position = config["probability_grid"]["min_position"] + indices * config["probability_grid"]["step_sizes"]
        position = np.concatenate((position,np.array([0.])))
        return torch.tensor(list(position))


def preprocess_text(text,objects,history_dict,tello):
    text = text.lower()
    text = text.replace(",","").replace('.',"").replace("fly","").replace('slide','').replace("flight","")
    anchor_object = None
    for key in objects:
        if key in text:
            anchor_object = key
            text = text.replace(key,'cube')

    if "go down" in text:
        tello.land()
        tello.end()
        history_dict["terminated"] == True
    return text,anchor_object
    

def transcribe_audio(audio_file):
    myarray = np.load(audio_file)
    audio_file = audio_file.replace('npy','wav')
    wavio.write(audio_file, myarray, 44100 ,sampwidth=1)

    speech_key, service_region = '4203927a90be4b1785cee6bdd8310f48', "eastus"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    audio_input = speechsdk.audio.AudioConfig(filename=audio_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    result = speech_recognizer.recognize_once()
    return result.text


def load_converter(corners_no_fly_zone,config):
    min_pos = torch.tensor([0.2,0.2,0.4,0.])
    steps = torch.tensor([0.2,0.2,0.2,0.25])
    max_pos = torch.tensor([4.6,4.6,1.8,0.75])
    number_poses = 16928
    converter = Converter(min_pos,max_pos,steps,number_poses,corners_no_fly_zone)
    return converter

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
    pose_model.load_state_dict(torch.load(dir_path + '/../drone/networks/pose_networks/' + config["pose_model"],map_location=torch.device('cpu')))
    pose_model.eval()
   
    
    #ft_model = fasttext.load_model(dir_path + '/../testing_mdn/cc.en.300.bin')
    ft_model = None
    mdn_model = MixtureDensityNetwork(300,3,4,ft_model,'debug')
    mdn_model.load_state_dict(torch.load(dir_path + '/../drone/networks/Fasttext_mdn/' + config["target_predictor_model"],map_location=torch.device('cpu')))
    #mdn_model.eval()

    return pose_model,mdn_model

def make_directories(test_folder):
    os.mkdir(test_folder)
    os.mkdir(test_folder + '/images')
    os.mkdir(test_folder + '/recordings')
    os.mkdir(test_folder + '/trajectories')
    os.mkdir(test_folder + '/probabilities')
    os.mkdir(test_folder + '/combined')

def main():
    

    # Book keeping 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = json.load(open(dir_path + '/config.json', 'r'))
    dt_string = datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y")
    test_folder =  "{}/tests/{}_{}".format(dir_path,config["name"],dt_string)
    make_directories(test_folder)

    # Load internal map
    objects = config[config["room"]]["objects"]
    objects_no_fly = objects_to_no_fly(objects)
    converter = load_converter(objects_no_fly,config)
    G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../graph_network/graphs/room_spain_03_4_orientations.gpickle')
    
    #load model
    pose_model,mdn_model = load_model(dir_path,config)    

    with torch.no_grad():
        
        history_dict = perform_test(pose_model,mdn_model,config,converter,test_folder,dir_path,objects,G)

        dict_file = test_folder + '/history_dict'
        f = open(dict_file + '.txt','a')
        f.write(str(history_dict))
        f.close()
        np.save(dict_file+'.npy', history_dict)

if __name__ == "__main__":
    main()



