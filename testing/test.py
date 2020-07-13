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

sys.path.append(os.path.abspath("/scratches/robot_2/fml35/mphil_project/navigation/testing"))
from models import Combi_Model
#from render_pose import render_pose
from visualisations import plot_trajectory,plot_action_predictions,visualise_poses

sys.path.append(os.path.abspath("/scratches/robot_2/fml35/mphil_project/navigation"))
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
        if converter.check_flyable_pose(pose.cuda()):
            break
    return pose

def load_image(folder_path,name):
    image = Image.open(folder_path +'/images/' + name)
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0).cuda()
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
        test_pose = true_pose.clone().cuda() + converter.move_to_coords[test_move]
        if converter.validate_pose(converter.map_pose(test_pose)):
            move = test_move
            break
    return move,action_predictions[0]

def perform_test(pose_model,action_predictor,lstm_model,text_dataset,config,converter,test_folder,dir_path,s):
    
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
    history_dict['target_indices'] = np.zeros((config["max_moves"]))
    history_dict['terminated'] = False

    constraints = []

    true_pose = sample_position(converter)
    target_pose = sample_position(converter)
    #target_pose = torch.tensor([1.1,0.3,0.5,0.5]).cuda()

    #target_name = render_pose(target_pose.cpu().numpy(),test_folder + '/images',-1)
    target_pose = target_pose.cpu().numpy().round(4)
    x,y,z,angle = target_pose[0],target_pose[1],target_pose[2],target_pose[3]
    target_name = 'render_-1_x_{:.4f}_y_{:.4f}_z_{:.4f}_rz_{:.4f}.png'.format(x,y,z,angle)
    history_dict['true_target'] = target_pose
    history_dict['target_name'] = target_name

    #subprocess.call([dir_path + '/render_shell.sh',str(x),str(y),str(z),str(angle),test_folder + '/images',target_name])
    s.sendall(pickle.dumps({'pose': target_pose,'path':test_folder + '/images','render_name':target_name}))
    status = s.recv(1024)

    predicted_target_pose = pose_model(load_image(test_folder,target_name))[0]
    predicted_target_pose[3] = predicted_target_pose[3] % 1.
    history_dict['predicted_targets'][history_dict["target_counter"]] = predicted_target_pose.cpu().numpy().round(4) 

    # Repeat until network says terminate or until have reached max_moves
    while history_dict['counter'] < config["max_moves"]:
        # Find next move
        #x,y,z,angle = true_pose.numpy().round(4)[0],true_pose.numpy().round(4)[1],true_pose.numpy().round(4)[2],true_pose.numpy().round(4)[3]
        current_name = 'render_{}_x_{:.4f}_y_{:.4f}_z_{:.4f}_rz_{:.4f}.png'.format(str(history_dict['counter']+1).zfill(2),x,y,z,angle)

        s.sendall(pickle.dumps({'pose': true_pose.numpy().round(4),'path':test_folder + '/images','render_name':current_name}))
        while True:
            data = s.recv(1024)
            if data == b'done':
                break

        #subprocess.call([dir_path + '/render_shell.sh',str(x),str(y),str(z),str(angle),test_folder + '/images',current_name])
        history_dict['image_names'].append(current_name)

        predicted_pose = pose_model(load_image(test_folder,current_name))[0]
        predicted_pose[3] = predicted_pose[3] % 1.
        history_dict['true_poses'][history_dict['counter']] = true_pose.cpu().numpy().round(4)
        history_dict['predicted_poses'][history_dict['counter']] = predicted_pose.cpu().numpy().round(4)

        print('Predicted Pose: {} Predicted Target: {}'.format(list(predicted_pose.cpu().numpy().round(3)),list(predicted_target_pose.cpu().numpy().round(3))))

        if (history_dict['counter']+1) % config["frequency_guidance"] == 0 and history_dict["counter"] > 1 and config["use_guidance"] == "True":
            
            plot_trajectory(history_dict, test_folder,converter,'end')
            plot_action_predictions(history_dict,test_folder,config)
            visualise_poses(history_dict,test_folder)

            #predicted_target_pose,constraints = describe_target(constraints,predicted_pose,predicted_target_pose)
            new_target_from_speech(test_folder,history_dict['target_counter'],lstm_model,text_dataset)
            
            
            history_dict['target_counter'] += 1
            history_dict['predicted_targets'][history_dict['target_counter']] = predicted_target_pose.cpu().numpy().round(4)
            plot_trajectory(history_dict, test_folder,converter,'start')
            history_dict['last_counter'] = history_dict['counter']
            
        history_dict['target_indices'][history_dict['counter']] = history_dict['target_counter']
        move,action_predictions = find_next_move(predicted_pose,predicted_target_pose,action_predictor,true_pose,converter)
        history_dict['action_predictions'][history_dict['counter']] = action_predictions.cpu().numpy().round(4)

        if move == 9:
            print('Terminate')
            #history_dict['terminated'] = True
            #break

        history_dict['counter'] +=1

        true_pose += converter.move_to_coords[move].cpu()
        true_pose = converter.map_pose(true_pose)

        # Check again that position is really allowed
        if  not converter.validate_pose(true_pose.cuda()):
            raise Exception("Outside of valid space")

    history_dict['statistics']['reached_target'] = ((torch.abs(true_pose - target_pose) < 0.05).all().item() and history_dict['terminated'] == True )


    return history_dict

def voice_command(path):
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

def new_target_from_speech(test_folder,counter,lstm_model,text_dataset):
    audio_file = voice_command(test_folder)
    text = transcribe_audio(audio_file)
    print(text)
    text,anchor_object = preprocess_text(text)
    length_description = len(text.split())
    vectorized_description = vectorize_text(text,text_dataset)

    output = lstm_model(None,vectorized_description.unsqueeze(0),length_description.unsqueeze(0))
    softmax = nn.Softmax(dim=2)
    output = softmax(output.reshape((-1,9*9*9,2)))
    _,index = torch.max(output[:,:,0])
    print(index)

def preprocess_text(text):
    text = text.lower()
    anchor_object = None
    if 'sofa' in text:
        anchor_object = 'sofa'
        text.replace('sofa','cube')
    elif 'armchair' in text:
        anchor_object = 'armchair'
        text.replace('armchair','cube')
    else:
        print('No anchor object detected')
    return text,anchor_object

    
def vectorize_text(text,text_dataset):
    vectorized_description = torch.zeros((1,40),dtype=torch.long).cuda()
    for j,word in enumerate(text.split()):
        vectorized_descriptions[0,j] = text_dataset.vocab.index(word)
    return vectorized_description
    





def transcribe_audio(audio_file):
    myarray = np.load(audio_file)
    audio_file = audio_file.replace('npy','wav')
    wavio.write(audio_file, myarray, 44100 ,sampwidth=1)

    audio_input = speechsdk.audio.AudioConfig(filename=audio_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    result = speech_recognizer.recognize_once()
    print(result.text)
    return result.text


def main():
    speech_key, service_region = '4203927a90be4b1785cee6bdd8310f48', "eastus"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)




def adjust_target(target_adjuster,predicted_target_pose):

    while True:
        try:
            text_instruction = input('Please tell me where to go: ')
            instruction = instruction_to_index(text_instruction)
        except KeyError:
            print("Please enter a valid instruction. Must be one of Stay, Go forward, Go backward,\nGo left, Go right, Go up, Go down, Rotate cw, Rotate acw, Keep going")
            continue
        else:
            break
    if instruction == 9:
        return predicted_target_pose
    else:
        one_hot_instruction = torch.zeros((9,)).cuda()
        one_hot_instruction[instruction] = 1.
        target_diff_vector = target_adjuster(torch.cat((predicted_target_pose,one_hot_instruction)).unsqueeze(0))[0]

        return predicted_target_pose + target_diff_vector

def describe_target(constraints,current_pose,predicted_target_pose):

    objects = {'armchair':np.array([-0.7,1.1,0.46,0.6,0.8,0.46]),
                'sofa': np.array([1.1,0.3,0.5,0.55,0.8,0.4])}

    space_dim = ((-1.6,2.2),(-0.8,1.7),(0.2,1.7))
    while True:
        try:
            text_instruction = input('Describe the pose: ')
            object_name,preposition = filter_text(text_instruction)
        except NameError:
            print("Please enter a valid description. Must contain sofa or armchair and left, right, front, behind, above or below.")
            continue
        else:
            break
    if object_name == 'no constraint':
        return predicted_target_pose,constraints
    else:
        constraints.append({'object_name':object_name,'preposition':preposition})
        current_pose_copy = current_pose.cpu().numpy()[:3]
        point = find_point(objects,current_pose_copy,constraints,space_dim)
        point.append(current_pose[3].item())
        return torch.tensor(point).cuda(),constraints

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


def load_converter():
    # min_pos = torch.tensor([-1.3,-0.5,0.2,0.]).cuda()
    # max_pos = torch.tensor([1.8,1.4,1.7,0.9375]).cuda()
    min_pos = torch.tensor([-1.6,-0.8,0.2,0.]).cuda()
    max_pos = torch.tensor([2.2,1.7,1.7,0.9375]).cuda()
    steps = torch.tensor([0.1,0.1,0.1,0.0625]).cuda()
    number_poses = 163840
    corners_no_fly_zone = torch.tensor([[[0.5,-0.5,0.1],[1.7,1.1,0.9]],[[-1.3,0.5,0.1],[-0.1,1.7,1.1]]]).cuda()
    converter = Converter(min_pos,max_pos,steps,number_poses,corners_no_fly_zone)
    return converter

def load_model(dir_path,config):

    pretrained_model = resnet18(pretrained=True)
    pretrained_model.fc = nn.Sequential()
    pose_model = Pose_Model(pretrained_model,512)
    pose_model.load_state_dict(torch.load(dir_path + '/../pose/experiments/' + config["pose_model"]))
    pose_model.eval().cuda()

    action_predictor = Decoder_Basic(10)
    action_predictor.load_state_dict(torch.load(dir_path + '/../action_predictor/experiments/' + config["action_predictor_model"] + '/checkpoints/last_epoch_model.pth'))
    action_predictor.eval().cuda()

    # target_adjuster = Target_Adjuster()
    # target_adjuster = nn.DataParallel(target_adjuster)
    # target_adjuster.load_state_dict(torch.load(dir_path + '/../target_adjuster/experiments/' + config["target_adjuster_model"] + '/checkpoints/last_epoch_model.pth'))
    # target_adjuster.eval().cuda()
    dataset = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new.csv',250)
    lstm_model = LSTM_model(32,5,dataset.len_vocab)


    return pose_model,action_predictor,LSTM_model,dataset

def main():
    print('Begin testing...')
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65433       # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT)) 
        print('Connected to server')   
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = json.load(open(dir_path + '/config.json', 'r'))
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

        converter = load_converter()
        
        #load model
        pose_model,action_predictor,lstm_model,text_dataset = load_model(dir_path,config)

        with torch.no_grad():
            for j in range(config["number_tests"]):

                dt_string = datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y")
                test_folder =  "{}/tests/{}_{}".format(dir_path,config["name"],dt_string)
                os.mkdir(test_folder)
                os.mkdir(test_folder + '/images')

                history_dict = perform_test(pose_model,action_predictor,lstm_model,text_dataset,config,converter,test_folder,dir_path,s)
                plot_trajectory(history_dict, test_folder,converter,'final')
                plot_action_predictions(history_dict,test_folder,config)
                visualise_poses(history_dict,test_folder)

                dict_file = test_folder + '/history_dict'
                f = open(dict_file + '.txt','a')
                f.write(str(history_dict))
                f.close()
                np.save(dict_file+'.npy', history_dict)

if __name__ == "__main__":
    main()


