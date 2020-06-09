import json
import numpy as np
import torch
import os
import shutil
from torchvision.models import vgg11, resnet18

def load_config(path):
    with open(path, 'r') as file:
        config = json.load(file)

        if config["data"]['place'] == "department":
            config["data"]["min_pos"] = torch.tensor([-1.4, -5.2, 1.3]).cuda()
            config["data"]["max_pos"] = torch.tensor([0.5, -3.3, 1.8]).cuda()
            config["data"]["number_images"]
            assert True == False, "Need to put number of images "
            config["sampler"]["no_fly_zone"] = None

        elif config["data"]["place"] == "living_room":
            config["data"]["min_pos"] = torch.tensor([-1.3, -0.5, 0.2,0.]).cuda()
            config["data"]["max_pos"] = torch.tensor([1.8, 1.4, 1.7,270.]).cuda()
            if config["sampler"]["no_fly"] == "True":
                config["sampler"]["no_fly_zone"] = torch.tensor([[[0.5,-0.5,0.2],[1.7,1.1,0.9]],[[-1.3,0.5,0.1],[-0.1,1.7,1.1]]]).cuda()
            elif config["sampler"]["no_fly"] == "False":
                config["sampler"]["no_fly_zone"] = None
    
    return config


def write_to_file(path,content):
    log_file = open(path,'a')
    log_file.write(content)
    log_file.flush()
    log_file.close()

def create_directories(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path + '/visualisations')
    os.mkdir(exp_path + '/visualisations/poses')
    os.mkdir(exp_path + '/visualisations/history')
    os.mkdir(exp_path + '/checkpoints')
    shutil.copy(exp_path +'/../../config.json',exp_path +'/config.json')
    shutil.copytree(exp_path +'/../../code',exp_path +'/code')
