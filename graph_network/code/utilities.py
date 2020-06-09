import json
import torch
import os
import shutil

from models.model import Graph_Replacer
from data.dataset import Graph_Replacer_Dataset
from data.sampler import Sampler


def load_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def load_model(config):
    if config["model"]["class"] == 'Graph_Replacer':
        return Graph_Replacer()

def load_data_set_and_sampler(config,exp_path):
    if config["graph"]["place"] == 'living_room':
        if config["graph"]["no_fly"] == "True": 
            dataset = Graph_Replacer_Dataset(exp_path + '/../../graphs/no_fly_living_room_small_angles.gpickle',config)
            sampler = Sampler(exp_path + '/../../graphs/no_fly_living_room_small_angles.gpickle',config)
    return dataset,sampler

def write_to_file(path,content):
    log_file = open(path,'a')
    log_file.write(content)
    log_file.flush()
    log_file.close()


def create_directories(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path + '/visualisations')
    os.mkdir(exp_path + '/visualisations/single_pairs')
    os.mkdir(exp_path + '/visualisations/history')
    os.mkdir(exp_path + '/visualisations/hard_pairs')
    os.mkdir(exp_path + '/checkpoints')
    shutil.copy(exp_path +'/../../config.json',exp_path +'/config.json')
    shutil.copytree(exp_path +'/../../code',exp_path +'/code')
