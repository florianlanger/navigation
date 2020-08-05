import json
import numpy as np
import torch
import os
import shutil
from torchvision.models import vgg11, resnet18

def load_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
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
