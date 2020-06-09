import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import torch.nn as nn
import os
import shutil

from models.basic_model import Net
from models.encoder import Encoder
from losses.basic_losses import calc_individual_loss_dot_product, calc_individual_loss_kl_div
from torchvision.models import vgg11, resnet18

def load_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    if config["data"]["orientation"] == "True":
        config["data"]["number_images"] = 9600
        config["data"]["dim_position"] = 4
        config["data"]["orientation"] == True
    elif config["data"]["orientation"] == "False":
        assert config["model"]["number_outputs"] == 8, "Need to have 8 outputs when no orientation"
        config["data"]["number_images"] = 2400
        config["data"]["dim_position"] = 3
        config["data"]["orientation"] == False
        
    if "place" in config["data"]:
        if config["data"]['place'] == "department":
            config["data"]["states_each_dimension"] = [20,20,6,4]
            config["data"]["min_pos"] = torch.tensor([-1.4, -5.2, 1.3,0.]).cuda()
            config["data"]["max_pos"] = torch.tensor([0.5, -3.3, 1.8,270.]).cuda()
            config["data"]["steps"] = torch.tensor([0.1,0.1,0.1,90.]).cuda()
            config["data"]["no_fly_zone"] = None
            config["graph"]["no_fly"] = False

        elif config["data"]["place"] == "living_room":
            config["data"]["states_each_dimension"] = [32,20,16,4]
            config["data"]["min_pos"] = torch.tensor([-1.3, -0.5, 0.2,0.]).cuda()
            config["data"]["max_pos"] = torch.tensor([1.8, 1.4, 1.7,270.]).cuda()
            config["data"]["steps"] = torch.tensor([0.1,0.1,0.1,90.]).cuda()
            if config["graph"]["no_fly"] == "True":
                config["data"]["no_fly_zone"] = torch.tensor([[0.5,-0.5,0.2,0.0],[1.7,1.1,0.9,270.]]).cuda()
                config["graph"]["no_fly"] = True
            elif config["graph"]["no_fly"] == "False":
                config["data"]["no_fly_zone"] = None
                config["graph"]["no_fly"] = False

            config["data"]["number_images"] = 40960
    #assert config["graph"]["type"] == "all_adjacent" or config["graph"]["type"] == "only_forward","Graph type must be 'all_adjacent' or 'only_forward'" 
    return config

def load_model(config):
    if config["model"]["class"] == 'Net':
        return Net(config["model"]["number_outputs"])
    elif config["model"]["class"] == 'Resnet':
        pretrained_model = resnet18(pretrained=True)
        pretrained_model.fc = nn.Sequential()
        return Encoder(pretrained_model,512,config["model"]["number_outputs"])
    elif config["model"]["class"] == 'VGG':
        pretrained_model = vgg11(pretrained=True)
        pretrained_model.classifier = pretrained_model.classifier[:-1]
        return Encoder(pretrained_model,4096,config["model"]["number_outputs"])



def write_to_file(path,content):
    log_file = open(path,'a')
    log_file.write(content)
    log_file.flush()
    log_file.close()


def update_checkpoint_epochs(epoch,loss,kind,exp_path):
    path = exp_path + '/checkpoints/model_info.txt'
    with open(path, 'r') as file:
        data = file.readlines()
    if kind == 'train':
        data[0] = 'Saved model with smallest train loss after epoch {} with loss {:.4f}\n'.format(epoch,loss)
    if kind == 'val':
        data[1] = 'Saved model with smallest validation loss after epoch {} with loss {:.4f}\n'.format(epoch,loss)

    with open(path, 'w') as file:
        file.writelines( data )

def update_accuracies(acc_individual,positions,accuracies,config):
    for i in range(acc_individual.shape[0]):
        # if j = 0 update statistics for position, if j =1 update for target
        for j in range(2):
            indices= position_to_array_indices(positions[i,j*4:(j+1)*4],config)
            i1,i2,i3,i4 = indices[0],indices[1],indices[2],indices[3]
            accuracies[j,i1,i2,i3,i4,0] += 1
            if accuracies[j,i1,i2,i3,i4,0] == 1:
                accuracies[j,i1,i2,i3,i4,1] = acc_individual[i]
            else:
                accuracies[j,i1,i2,i3,i4,1] = 1/accuracies[j,i1,i2,i3,i4,0] * acc_individual[i] + (accuracies[j,i1,i2,i3,i4,0]-1)/ accuracies[j,i1,i2,i3,i4,0] *accuracies[j,i1,i2,i3,i4,1]

def update_counters_actions(action_penalties,output,counters_actions):
    _,indices = torch.max(output,dim=1)
    for index in indices:
        counters_actions[0,index] += 1
    counters_actions[1] += torch.sum((action_penalties < 0.0001).to(torch.float16)/ torch.sum(action_penalties < 0.0001,dim=1).view(-1,1),dim=0)

def position_to_array_indices(position,config):
    indices = list( ((position - config["data"]["min_pos"]) / config["data"]["steps"]).round().cpu().numpy().astype(int) )
    return indices


def create_directories(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path + '/visualisations')
    os.mkdir(exp_path + '/visualisations/single_pairs')
    os.mkdir(exp_path + '/visualisations/heatmaps')
    os.mkdir(exp_path + '/visualisations/trajectories')
    os.mkdir(exp_path + '/visualisations/history')
    os.mkdir(exp_path + '/visualisations/frequencies')
    os.mkdir(exp_path + '/visualisations/hard_pairs')
    os.mkdir(exp_path + '/checkpoints')
    shutil.copy(exp_path +'/../../config.json',exp_path +'/config.json')
    shutil.copytree(exp_path +'/../../code',exp_path +'/code')


# acc_individual = torch.tensor([True,True,True,False]).cuda()
# positions = torch.tensor([[-1.0,-4.9,1.4,-1.3,-5.2,1.3],[-1.0,-4.9,1.4,-1.2,-5.2,1.3],[-1.0,-4.9,1.4,-1.3,-5.0,1.3],[-1.0,-4.9,1.4,-1.3,-5.0,1.3]]).cuda()
# accuracies = torch.zeros(2,20,20,6,2).cuda()
# update_accuracies(acc_individual,positions,accuracies)