import json
import torch
import os
import shutil

from models.decoder import Decoder_Basic
from data.decoder_data import Decoder_Dataset
from data.sampler import Sampler


def load_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config


def load_model(config):
    if config["model"]["class"] == 'Decoder_Basic':
        return Decoder_Basic(10)


def load_data_set_and_sampler(config,exp_path):
    if config["graph"]["place"] == 'department':
        dataset = Decoder_Dataset(None,config)
    elif config["graph"]["place"] == 'living_room':
        if config["graph"]["no_fly"] == "False":
            dataset = Decoder_Dataset(None,config)
        elif config["graph"]["no_fly"] == "True": 
            dataset = Decoder_Dataset(exp_path + '/../../../graph_network/graphs/no_fly_living_room.gpickle',config)
            sampler = Sampler(exp_path + '/../../../graph_network/graphs/no_fly_living_room.gpickle',config)
    return dataset,sampler




def write_to_file(path,content):
    log_file = open(path,'a')
    log_file.write(content)
    log_file.flush()
    log_file.close()



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

def update_counters_actions(target_distributions,output,counters_actions):
    _,indices = torch.max(output,dim=1)
    for index in indices:
        counters_actions[0,index] += 1
    counters_actions[1] += torch.sum((target_distributions < 0.0001).to(torch.float16)/ torch.sum(target_distributions < 0.0001,dim=1).view(-1,1),dim=0)

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
