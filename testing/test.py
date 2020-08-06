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
import socket
import pickle

import sounddevice as sd
from scipy.io.wavfile import write
import wavio
import azure.cognitiveservices.speech as speechsdk

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation/testing"))

from visualisations import plot_trajectory,plot_action_predictions,visualise_poses,plot_global_probability, plot_combined, plot_trajectory_1

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation"))
from graph_network.code.graphs.conversions import Converter
from action_predictor.code.models.decoder import Decoder_Basic
from pose.code.models.model import Pose_Model
from target_adjuster.code.models.model import Target_Adjuster
from target_pose.target_pose import find_point,filter_text
from train_target_predictor.code.data import Target_Predictor_Dataset
from train_target_predictor.code.model import LSTM_model



def sample_position(converter):
    while True:
        position = (converter.max_position.cpu() - converter.min_position.cpu())[:3] * torch.rand((3,)) + converter.min_position.cpu()[:3]
        pose = torch.cat((position,torch.rand((1,))))
        if converter.check_flyable_pose(pose): #.cuda()):
            break
    return pose

def load_image(folder_path,name):
    image = Image.open(folder_path +'/images/' + name)
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0)#.cuda()
    return image

def find_next_move(predicted_pose,predicted_target_pose,action_predictor,true_pose,converter):
    action_predictions = action_predictor(torch.cat((predicted_pose,predicted_target_pose),dim=0).unsqueeze(0))
    # moves recommended by networking ordered by descending probability
    _,move_numbers = torch.sort(action_predictions[0],descending=True)
    # Go through moves in order how they were recommended.
    # If move is allowed and has not been done in this pos before, break and do the move
    move = None
    for test_move in move_numbers:
        test_move = test_move.item()
        test_pose = true_pose.clone() + converter.move_to_coords[test_move]     #.cuda() + converter.move_to_coords[test_move]
        if converter.validate_pose(converter.map_pose(test_pose)):
            move = test_move
            break
    return move,action_predictions[0]

def perform_test(pose_model,action_predictor,lstm_model,text_dataset,config,converter,test_folder,dir_path,s,objects):
    
    history_dict = {}
    history_dict['statistics'] = {}
    history_dict['true_poses'] = np.zeros((config["max_moves"],4))
    history_dict['predicted_poses'] = np.zeros((config["max_moves"],4))
    history_dict['predicted_targets'] = np.zeros((config["max_moves"],4))
    history_dict['action_predictions'] = np.zeros((config["max_moves"],10))
    history_dict['image_names'] = []
    history_dict['counter'] = 0
    history_dict['last_counter'] = 0
    history_dict['target_counter'] = 0
    history_dict['terminated'] = False
    history_dict['global_probabilities'] = np.zeros(config["probability_grid"]["points_per_dim"])


    true_pose = sample_position(converter)

    # Repeat until network says terminate or until have reached max_moves
    while history_dict['counter'] <  config["max_moves"]:
        # Find next move
        x,y,z,angle = true_pose.numpy().round(4)[0],true_pose.numpy().round(4)[1],true_pose.numpy().round(4)[2],true_pose.numpy().round(4)[3]
        current_name = 'render_{}_x_{:.4f}_y_{:.4f}_z_{:.4f}_rz_{:.4f}.png'.format(str(history_dict['counter']).zfill(2),x,y,z,angle)
        
        # BLENDER
        s.sendall(pickle.dumps({'pose': true_pose.numpy().round(4),'path':test_folder + '/images','render_name':current_name}))
        while True:
            data = s.recv(1024)
            if data == b'done':
                break

        history_dict['image_names'].append(current_name)

        # BLENDER
        predicted_pose = pose_model(load_image(test_folder,current_name))[0]
        #predicted_pose = torch.tensor([0.,1.,1.,0.5])

        predicted_pose[3] = predicted_pose[3] % 1.
        history_dict['true_poses'][history_dict['counter']] = true_pose.cpu().numpy().round(4)
        history_dict['predicted_poses'][history_dict['counter']] = predicted_pose.cpu().numpy().round(4)

        #if history_dict['counter'] == 0 or history_dict['counter'] == 4:
        if True:
            contribution_global_probabilities,text,anchor_object = update_target_probabilities_from_speech(test_folder,history_dict['target_counter'],lstm_model,text_dataset,objects,config)
            history_dict['global_probabilities'] = history_dict['global_probabilities'] + contribution_global_probabilities
            predicted_target_pose = find_current_target(history_dict["global_probabilities"],config)
            history_dict['predicted_targets'][history_dict['target_counter']] = predicted_target_pose.cpu().numpy().round(4)
            history_dict['last_counter'] = history_dict['counter']
            plot_global_probability(history_dict['global_probabilities'],test_folder,converter,config,history_dict['target_counter'],'total')
            plot_global_probability(contribution_global_probabilities,test_folder,converter,config,history_dict['target_counter'],'last',text,anchor_object=anchor_object)
            history_dict['target_counter'] += 1
            
        print('Predicted Pose: {} Predicted Target: {}'.format(list(predicted_pose.cpu().numpy().round(3)),list(predicted_target_pose.cpu().numpy().round(3))))

        move,action_predictions = find_next_move(predicted_pose,predicted_target_pose,action_predictor,true_pose,converter)
        history_dict['action_predictions'][history_dict['counter']] = action_predictions.cpu().numpy().round(4)

        #create visualisations
        plot_trajectory(history_dict, test_folder,converter)
        plot_action_predictions(history_dict,test_folder,config,converter)
        visualise_poses(history_dict,test_folder)
        plot_combined(history_dict,test_folder)
        
        # #make pop up window
        # img = Image.open('{}/combined/combined_{}.png'.format(test_folder,str(history_dict["counter"]).zfill(2)))
        # img.show()
        

        if move == 9:
            print('Terminate')

        history_dict['counter'] +=1

        true_pose += converter.move_to_coords[move].cpu()
        true_pose = converter.map_pose(true_pose)

        # Check again that position is really allowed
        if  not converter.validate_pose(true_pose): #.cuda()):
            raise Exception("Outside of valid space")

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




def update_target_probabilities_from_speech(test_folder,counter,lstm_model,text_dataset,objects,config):
    audio_file = voice_command(test_folder,counter)
    original_text = transcribe_audio(audio_file)

    #original_text = config[config["room"]]["scenarios_commands"][str(config['scenario'])][counter]
    text,anchor_object = preprocess_text(original_text,objects)
    length_description = torch.tensor([len(text.split())])
    vectorized_description, new_anchor_object = vectorize_text(text,text_dataset,objects)
    if new_anchor_object is not None:
        anchor_object = new_anchor_object


    output = lstm_model(None,vectorized_description.unsqueeze(0),length_description)

    softmax = nn.Softmax(dim=1)
    output = softmax(output.reshape((9*9*9,2)))


    # determine new target
    # _,index = torch.max(output[:,0].flatten(),dim=0)
    # index = index.item()
    # indices = np.array([index // 81,  (index % 81)  // 9, index % 9])
    # print(indices)
    # position = objects[anchor_object]['dimensions'][:3] + (indices - 4) * objects[anchor_object]['dimensions'][3:6]
    # position = np.concatenate((position,np.array([0.])))
    # position = torch.tensor(list(position))

    # find contribution to gloabl probability distribution
    probability_contribution = map_local_probabilities_to_global(output[:,0],objects[anchor_object],config)

    return probability_contribution,original_text,anchor_object


def find_current_target(global_probabilities,config):
        points_per_dim = config["probability_grid"]["points_per_dim"]
        index = np.argmax(global_probabilities.flatten())
        index = index.item()
        indices = np.array([index // (points_per_dim[1]*points_per_dim[2]),  (index % (points_per_dim[1]*points_per_dim[2]))  // points_per_dim[1], index % points_per_dim[2]])
        position = config["probability_grid"]["min_position"] + indices * config["probability_grid"]["step_sizes"]
        position = np.concatenate((position,np.array([0.])))
        return torch.tensor(list(position))

# def map_local_probabilities_to_global(output,object_dims,config):
#     points_per_dim = config["probability_grid"]["points_per_dim"]
#     probabilities = np.zeros(points_per_dim)
#     for i in range(9):
#         for j in range(9):
#             for k in range(9):
#                 probability = output[81*i + 9*j + k].item()
#                 g1,g2,g3 = find_global_indices_from_local_indices(i,j,k,object_dims,config)
#                 if g1 >= 0 and g2 >= 0 and g3 >= 0 and g1 < points_per_dim[0] and g2 < points_per_dim[1] and g3 < points_per_dim[2]:
#                     probabilities[g1,g2,g3] = probability

#     return probabilities

# def find_global_indices_from_local_indices(l1,l2,l3,object_dims,config):
#     indices = np.array([l1,l2,l3])
#     indices -= 4
#     position = object_dims[:3] + indices * object_dims[3:6]
#     position = position.round(1)
#     new_indices = ((position - config["probability_grid"]["min_position"])/np.array(config["probability_grid"]["step_sizes"])).round().astype(int)
#     return new_indices[0],new_indices[1],new_indices[2]


def map_local_probabilities_to_global(output,anchor_object,config):
    points_per_dim = config["probability_grid"]["points_per_dim"]
    probabilities = np.zeros(points_per_dim)
    for i in range(points_per_dim[0]):
        for j in range(points_per_dim[1]):
            for k in range(points_per_dim[2]):

                #probability = output[81*i + 9*j + k].item()
                local_indices = find_local_indices_from_global_indices(i,j,k,anchor_object,config)
                if (local_indices > 4).any() or (local_indices < -4).any():
                    pass
                else:
                    local_indices += 4
                    probabilities[i,j,k] = output[81*local_indices[0] + 9*local_indices[1] + local_indices[2]].item()

    return probabilities

def find_local_indices_from_global_indices(g1,g2,g3,anchor_object,config):
    indices = np.array([g1,g2,g3])
    position = config["probability_grid"]["min_position"] + indices * config["probability_grid"]["step_sizes"]
    #position = object_dims[:3] + indices * object_dims[3:6]
    print()
    print(np.array(anchor_object["scaling"]))
    print(np.array(anchor_object["dimensions"][3:6]))
    local_indices = ((position - np.array(anchor_object["dimensions"][:3]))/(np.array(anchor_object["scaling"])*np.array(anchor_object["dimensions"][3:6]))).round().astype(int)
    return local_indices



def preprocess_text(text,objects):
    text = text.lower()
    text = text.replace(",","").replace('.',"")
    anchor_object = None
    for key in objects:
        if key in text:
            anchor_object = key
            text = text.replace(key,'cube')

    return text,anchor_object

    
def vectorize_text(text,text_dataset,objects):
    vectorized_description = torch.zeros((40),dtype=torch.long) #.cuda()
    split_text = text.split()
    j = 0
    anchor_object = None
    while j < len(split_text):
        try:
            vectorized_description[j] = text_dataset.vocab.index(split_text[j])
            j += 1
        except ValueError:
            text_replacement = input('Replace {} with: '.format(split_text[j]))
            for key in objects:
                if text_replacement == key:
                    split_text[j] = 'cube'
                    anchor_object = text_replacement
                    break
            else:
                split_text[j] = text_replacement
    return vectorized_description,anchor_object
    


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
    # min_pos = torch.tensor([-1.3,-0.5,0.2,0.]).cuda()
    # max_pos = torch.tensor([1.8,1.4,1.7,0.9375]).cuda()
    min_pos = torch.tensor([0.2,0.2,0.4,0.]) #.cuda()
    steps = torch.tensor([0.1,0.1,0.1,0.0625]) #.cuda()
    if config["room"] == 'my_room':
        max_pos = torch.tensor([4.0,3.4,1.8,0.9375]) #.cuda()
        number_poses = 272384
    elif config["room"] == 'room_books':
        max_pos = torch.tensor([4.0,4.8,1.8,0.9375])
        number_poses = 391552
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
    pose_model.load_state_dict(torch.load(dir_path + '/../drone/networks/pose.pth',map_location=torch.device('cpu')))
    #pose_model.load_state_dict(torch.load(dir_path + '/../pose/experiments/' + config["pose_model"]))
    pose_model.eval()

    action_predictor = Decoder_Basic(10)
    action_predictor.load_state_dict(torch.load(dir_path + '/../drone/networks/exp_6_small_angles_hard_pairs_time_15_02_48_date_02_06_2020_last_epoch_model.pth',map_location=torch.device('cpu')))
    #action_predictor.load_state_dict(torch.load(dir_path + '/../action_predictor/experiments/' + config["action_predictor_model"] + '/checkpoints/last_epoch_model.pth'))
    action_predictor.eval()

    dataset = Target_Predictor_Dataset(dir_path + '/../target_pose/training_data1/data_transcribed_new.csv',250)
    lstm_model = LSTM_model(5,32,63)
    lstm_model.load_state_dict(torch.load(dir_path + '/../drone/networks/debug_testing_time_14_44_52_date_15_07_2020_epoch_241_model.pth',map_location=torch.device('cpu')))
    lstm_model.eval()

    return pose_model,action_predictor,lstm_model,dataset

def main():
    # BLENDER
    print('Begin testing...')
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65437   # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT)) 
        print('Connected to server')
        #s = 'dummy'


        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = json.load(open(dir_path + '/config.json', 'r'))
        #os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

        objects = config[config["room"]]["objects"]

        
        objects_no_fly = objects_to_no_fly(objects)

        converter = load_converter(objects_no_fly,config)
        
        #load model
        pose_model,action_predictor,lstm_model,text_dataset = load_model(dir_path,config)

        with torch.no_grad():
            for j in range(config["number_tests"]):

                dt_string = datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y")
                test_folder =  "{}/tests/{}_{}".format(dir_path,config["name"],dt_string)
                os.mkdir(test_folder)
                os.mkdir(test_folder + '/images')
                os.mkdir(test_folder + '/recordings')
                os.mkdir(test_folder + '/action_predictions')
                os.mkdir(test_folder + '/poses')
                os.mkdir(test_folder + '/trajectories')
                os.mkdir(test_folder + '/probabilities')
                os.mkdir(test_folder + '/combined')

                history_dict = perform_test(pose_model,action_predictor,lstm_model,text_dataset,config,converter,test_folder,dir_path,s,objects)

                dict_file = test_folder + '/history_dict'
                f = open(dict_file + '.txt','a')
                f.write(str(history_dict))
                f.close()
                np.save(dict_file+'.npy', history_dict)

if __name__ == "__main__":
    main()



