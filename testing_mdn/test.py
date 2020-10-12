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
import fasttext
import select
import networkx as nx

# import sounddevice as sd
# from scipy.io.wavfile import write
# import wavio
# import azure.cognitiveservices.speech as speechsdk

sys.path.append(os.path.abspath("/scratches/robot_2/fml35/mphil_project/navigation"))

from visualisations import plot_trajectory,plot_action_predictions,visualise_poses,plot_global_probability, plot_combined, plot_trajectory_1

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation"))
from graph_network.code.graphs.conversions import Converter
from action_predictor.code.models.decoder import Decoder_Basic
from pose.code.models.model import Pose_Model
from target_adjuster.code.models.model import Target_Adjuster
from target_pose.target_pose import find_point,filter_text
from train_target_predictor_03_mdn.code.model import MixtureDensityNetwork


def sample_position(converter):
    while True:
        position = (converter.max_position.cpu() - converter.min_position.cpu())[:3] * torch.rand((3,)) + converter.min_position.cpu()[:3]
        pose = torch.cat((position,torch.rand((1,))))
        if converter.check_flyable_pose(pose):
            break
    return pose

def load_image(folder_path,name):
    image = Image.open(folder_path +'/images/' + name)
    #image = Image.open('/Users/legend98/Desktop/render_2043_x_2.318_y_1.633_z_1.283_rz_35.94.png')
    image = F.to_tensor(image)[:3,:,:].unsqueeze(0)#.cuda()
    return image


def find_next_move_graph(start_pose,target_pose,converter,G,number_of_moves):

    convert_action_to_number = {"pos x": 1, "neg x":2, "pos y":3,"neg y":4,"pos z":5,"neg z":6,"rot +":7,"rot -":8}

    #noise = torch.cat((0.4*torch.rand(3)-0.2,torch.zeros((1,))))
    #noise = torch.cat((torch.zeros(3,),0.1666*torch.ones((1,))-0.8333))
    noise = torch.cat((0.4*torch.rand(3)-0.2,0.1666*torch.ones((1,))-0.0833))
    if converter.check_flyable_pose(start_pose + noise):
        start_pose += noise

    changes = torch.tensor([[0.,0.,0.,0],[0.2,0.,0.,0],[-0.2,0.,0.,0],[0.,0.2,0.0,0],[0.,-0.2,0.,0],[0.,0.,0.2,0],[0.,0.,-0.2,0]])
    counter = 0

    print('start pose',start_pose)
    print('flyable',converter.check_flyable_pose(start_pose))

    while not converter.check_flyable_pose(start_pose):
        print('trying changes so that position is valid')
        start_pose -= changes[counter]
        start_pose += changes[counter+1]
        counter +=1


    index_start = converter.pose_to_index(start_pose)
    index_end = converter.pose_to_index(target_pose)
    shortest_path = nx.shortest_path(G,index_start,index_end)
    
    l = len(shortest_path)

    if l == 1:
        return [9] * number_of_moves
    else:
        actions = []
        for i in range(l-1):
            actions.append(G[shortest_path[i]][shortest_path[i+1]][0]['action'])
        moves = [convert_action_to_number[action] for action in actions]

        # Fill with terminate actions
        if l -1 < number_of_moves:
            for i in range(number_of_moves + 1 -l):
                moves.append(9)
        return moves[:number_of_moves]
    

def find_next_move(predicted_pose,predicted_target_pose,action_predictor,true_pose,converter,history_dict):
    predicted_target_pose = predicted_target_pose.to(torch.float32)
    action_predictions = action_predictor(torch.cat((predicted_pose,predicted_target_pose),dim=0).unsqueeze(0))
    # moves recommended by networking ordered by descending probability
    _,move_numbers = torch.sort(action_predictions[0],descending=True)
    # Go through moves in order how they were recommended.
    # If move is allowed and has not been done in this pos before, break and do the move
    #diff_conventions = {0:0,3:1,2:2,4:3,1:4,5:5,6:6,7:7,8:8,9:9}
    opposite_move = {0:0,1:2,2:1,3:4,4:3,5:6,6:5,7:8,8:7,9:9}

    move = None
    for test_move in move_numbers:
        test_move = test_move.item()
        test_pose = true_pose.clone() + converter.move_to_coords[test_move]
        if converter.validate_pose(converter.map_pose(test_pose)) and test_move != opposite_move[history_dict['actions'][history_dict['counter']-1]]:
            move = test_move
            break
    return move,action_predictions[0]

def perform_test(pose_model,action_predictor,mdn_model,config,converter,test_folder,dir_path,s,objects,scenario,max_number_descriptions,true_pose,G,number_of_moves_graph):
    print('start perform test')
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
    history_dict['target'] = config[config["room"]]["scenarios_target_positions"][str(scenario)]
    history_dict['actions'] = np.zeros(config["max_moves"])


    #x_2.318_y_1.633_z_1.283_rz_35.94

    # Repeat until network says terminate or until have reached max_moves

    while history_dict['counter'] < config["max_moves"]:    
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
        # if predicted_pose[2] < 0.3:
        #     print('predicted_pose before subtract', predicted_pose)
        #     predicted_pose[2] -= 0.2
        #     print('predicted_pose after subtract', predicted_pose)
        #predicted_pose[3] = torch.round(predicted_pose[3] * 4.) /4.

        history_dict['true_poses'][history_dict['counter']] = true_pose.cpu().numpy().round(4)
        history_dict['predicted_poses'][history_dict['counter']] = predicted_pose.cpu().numpy().round(4)

        #if history_dict['counter'] == 0 or history_dict['counter'] == 4:
        #if detected_keystroke == True:
        while history_dict["target_counter"] < max_number_descriptions:
            contribution_global_probabilities,text,anchor_object = update_target_probabilities_from_speech(test_folder,history_dict['target_counter'],mdn_model,objects,config,scenario)
            history_dict['global_probabilities'] = history_dict['global_probabilities'] + contribution_global_probabilities
            predicted_target_pose = find_current_target(history_dict["global_probabilities"],config,converter)
            #predicted_target_pose = torch.tensor(history_dict['target'])
            history_dict['predicted_targets'][history_dict['target_counter']] = predicted_target_pose.cpu().numpy().round(4)
            history_dict['last_counter'] = history_dict['counter']
            # plot_global_probability(history_dict['global_probabilities'],test_folder,converter,config,history_dict['target_counter'],'total',scenario)
            # plot_global_probability(contribution_global_probabilities,test_folder,converter,config,history_dict['target_counter'],'last',scenario,text,anchor_object=anchor_object)
            history_dict['target_counter'] += 1
            
        print('Predicted Pose: {} Predicted Target: {}'.format(list(predicted_pose.cpu().numpy().round(3)),list(predicted_target_pose.cpu().numpy().round(3))))

        #move,action_predictions = find_next_move(predicted_pose,predicted_target_pose,action_predictor,true_pose,converter,history_dict)
        if history_dict['counter'] % number_of_moves_graph == 0:
            moves = find_next_move_graph(predicted_pose,predicted_target_pose,converter,G,number_of_moves_graph)
            history_dict['actions'][history_dict['counter']:history_dict['counter']+number_of_moves_graph] = moves
        
        move = history_dict['actions'][history_dict['counter']]

        #create visualisations
        # plot_trajectory(history_dict, test_folder,converter)
        # plot_action_predictions(history_dict,test_folder,config,converter)
        # visualise_poses(history_dict,test_folder)
        # plot_combined(history_dict,test_folder)
        
        # #make pop up window
        # img = Image.open('{}/combined/combined_{}.png'.format(test_folder,str(history_dict["counter"]).zfill(2)))
        # img.show()
        

        if move == 9:
            print('Terminate')
            break

        history_dict['counter'] +=1

        true_pose += converter.move_to_coords[move].cpu()
        true_pose = converter.map_pose(true_pose)
        if true_pose[2] < 0.0:
            true_pose[2] = 0.03

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


def update_target_probabilities_from_speech(test_folder,counter,mdn_model,objects,config,scenario,tello=None):
    #audio_file = voice_command(test_folder,counter)
    #if tello is not None:
    #    tello.rotate_clockwise(1)
    #original_text = transcribe_audio(audio_file)

    original_text = config[config["room"]]["scenarios_commands"][str(scenario)][counter]
    text,anchor_object = preprocess_text(original_text,objects)
    pi,normal = mdn_model([text])#,torch.tensor([float(len(text.split()))]))

    probability_contribution = find_probability_contribution_from_parameters(pi,normal,objects[anchor_object],config)
    return probability_contribution,original_text,anchor_object

def find_probability_contribution_from_parameters(pi,normal,anchor_object,config):

    points_per_dim = config["probability_grid"]["points_per_dim"]
    global_probabilities = np.zeros((points_per_dim[0],points_per_dim[1],points_per_dim[2]))
    for i in range(points_per_dim[0]):
        for j in range(points_per_dim[1]):
            for k in range(points_per_dim[2]):
                #print(i,j,k)
               
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

def find_current_target(global_probabilities,config,converter):
        points_per_dim = config["probability_grid"]["points_per_dim"]
        indices_sorted = np.argsort(global_probabilities.flatten())[::-1]

        for index in indices_sorted:
            indices = np.array([index // (points_per_dim[1]*points_per_dim[2]),  (index % (points_per_dim[1]*points_per_dim[2]))  // points_per_dim[2], index % points_per_dim[2]])
            position = config["probability_grid"]["min_position"] + indices * config["probability_grid"]["step_sizes"]
            position = np.concatenate((position,np.array([0.])))
            if converter.check_flyable_pose(torch.from_numpy(position)):
                return torch.tensor(list(position))


def preprocess_text(text,objects):
    text = text.lower()
    text = text.replace(",","").replace('.',"").replace("fly","").replace('slide','').replace("flight","")
    anchor_object = None
    for key in objects:
        if key in text:
            anchor_object = key
            text = text.replace(key,'cube')

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

    min_pos = torch.tensor([-1.9,-1.,0.,0.])
    max_pos = torch.tensor([3.1,2.2,2.4,0.91666666])
    steps = torch.tensor([0.2,0.2,0.2,0.0833333])
    number_poses = 26 * 17 * 13 * 12

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
    #pose_model.load_state_dict(torch.load(dir_path + '/../pose/experiments/' + config["pose_model"]))
    pose_model.eval()

    action_predictor = Decoder_Basic(10)
    action_predictor.load_state_dict(torch.load(dir_path + '/../drone/networks/action_predictor/' + config["action_predictor_model"],map_location=torch.device('cpu')))
    #action_predictor.load_state_dict(torch.load(dir_path + '/../action_predictor/experiments/' + config["action_predictor_model"] + '/checkpoints/last_epoch_model.pth'))
    action_predictor.eval()

    #ft_model = fasttext.load_model(dir_path + '/cc.en.300.bin')
    ft_model = fasttext.load_model('/data/cvfs/fml35/original_downloads/Fasttext/cc.en.300.bin')
    #ft_model = None
    mdn_model = MixtureDensityNetwork(300,3,4,ft_model,'normal')
    mdn_model.load_state_dict(torch.load(dir_path + '/../drone/networks/Fasttext_mdn/' + config["target_predictor_model"],map_location=torch.device('cpu')))
    #mdn_model.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_04_fasttext_and_lstm/experiments/exp_01_time_17_44_05_date_12_08_2020/checkpoints/epoch_81_model.pth',map_location=torch.device('cpu')))
    #mdn_model.eval()

    return pose_model,action_predictor,mdn_model

def main():
    # BLENDER
    print('Begin testing...')
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65444   # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT)) 
        print('Connected to server')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = json.load(open(dir_path + '/config.json', 'r'))
        #os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

        objects = config[config["room"]]["objects"]

        
        objects_no_fly = objects_to_no_fly(objects)

        converter = load_converter(objects_no_fly,config)
        
        #load model
        print('loading models')
        pose_model,action_predictor,mdn_model = load_model(dir_path,config)
        dt_string = datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y")
        big_test_folder =  "{}/tests/{}".format(dir_path,dt_string)
        os.mkdir(big_test_folder)

        #load graph
        print('loading graph')
        G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../graph_network/graphs/ignas_big_room_wide_grid.gpickle')

        number_of_moves_graph = 1
        with open(big_test_folder + '/summary.txt','a') as summary:
            summary.write('name start_position predicted_target final_position target dist_to_predicted_target dist_final_position_to_target\n')

        with torch.no_grad():
            for scenario in range(3,7):
                #true_pose = sample_position(converter)
                true_pose = torch.tensor(config[config["room"]]["scenarios_start_positions"][str(scenario)])

                for max_number_descriptions in range(5,6):
                    name = "scenario_{}_descriptions_{}".format(str(scenario).zfill(2),max_number_descriptions)
                    test_folder =  "{}/tests/{}/{}".format(dir_path,dt_string,name)
                    os.mkdir(test_folder)
                    os.mkdir(test_folder + '/images')
                    os.mkdir(test_folder + '/recordings')
                    os.mkdir(test_folder + '/action_predictions')
                    os.mkdir(test_folder + '/poses')
                    os.mkdir(test_folder + '/trajectories')
                    os.mkdir(test_folder + '/probabilities')
                    os.mkdir(test_folder + '/combined')

                    history_dict = perform_test(pose_model,action_predictor,mdn_model,config,converter,test_folder,dir_path,s,objects,scenario,max_number_descriptions,true_pose.clone(),G,number_of_moves_graph)

                    dict_file = test_folder + '/history_dict'
                    f = open(dict_file + '.txt','a')
                    f.write(str(history_dict))
                    f.close()
                    np.save(dict_file+'.npy', history_dict)

                    with open(big_test_folder + '/summary.txt','a') as summary:
                        dist_to_predicted_target = np.linalg.norm((np.array(history_dict['predicted_targets'][history_dict['target_counter']-1]) - history_dict['target'])[:3]).round(4)
                        dist_final_position_to_target = np.linalg.norm((np.array(history_dict['true_poses'][history_dict['counter']-1]) - history_dict['target'])[:3]).round(4)
                        print(dist_to_predicted_target)
                        summary.write('{}#{}#{}#{}#{}#{}#{}\n'.format(name,history_dict['true_poses'][0],history_dict['predicted_targets'][history_dict['target_counter']-1],history_dict['true_poses'][history_dict['counter']-1],history_dict['target'],dist_to_predicted_target,dist_final_position_to_target))


if __name__ == "__main__":
    main()



