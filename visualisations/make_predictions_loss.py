import gensim.downloader as api
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
from datetime import datetime
from torch.utils import data
import fasttext
import fasttext.util
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

#sys.path.append(os.path.abspath("/scratches/robot_2/fml35/mphil_project/navigation/visualisations"))

from ablation_loss_visualisations import visualise
sys.path.append(os.path.abspath("/scratches/robot_2/fml35/mphil_project/navigation"))

from train_target_predictor_fasttext.code.model import Fasttext_model


from train_target_predictor_mdn.code.data import Target_Predictor_Dataset
from train_target_predictor_mdn.code.model import MixtureDensityNetwork
from train_target_predictor_mdn.code.utilities import calc_node_probabilities


def make_predictions(model,data_loader,kind,return_targets = False):

    if return_targets == True:
        targets = torch.zeros((len(data_loader.dataset),3))

    outputs = torch.zeros((len(data_loader.dataset),9*9*9))
    for i,(descriptions,target_poses) in enumerate(data_loader):

            output = model(descriptions)

            if kind == "crossentropy":
                softmax = nn.Softmax(dim=1)
                output = softmax(output.reshape((-1,9*9*9)))
                output_for_visualisation = output.detach().cpu()
                print(output_for_visualisation.shape)

            elif kind == "binary":
                softmax = nn.Softmax(dim=2)
                output = softmax(output.reshape((-1,9*9*9,2)))
                output_for_visualisation = output[:,:,0].detach().cpu()

            elif kind == "mdn":
                pi,normal = output
                output_for_visualisation = calc_node_probabilities(pi, normal)
            
            outputs[i*bs:(i+1)*bs,:] = output_for_visualisation
            if return_targets == True:
                targets[i*bs:(i+1)*bs,:] = target_poses.cpu()
    
    if return_targets == True:
        return outputs,targets
    else:
        return outputs



def save_predictions():
    bs = 25

    train_data = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_training.csv',200)
    val_data = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_validation.csv',50)
    train_loader = data.DataLoader(train_data, batch_size = bs, shuffle=False)
    val_loader = data.DataLoader(val_data, batch_size = bs,shuffle=False)


    ft_model = fasttext.load_model('/data/cvfs/fml35/original_downloads/Fasttext/cc.en.300.bin')
    print('finished loading fasttext model')

    model_crossentropy = Fasttext_model(300,ft_model,"True")
    model_crossentropy.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_fasttext/experiments/exp_03_proper_split_cross_entropy_time_18_00_37_date_11_08_2020/checkpoints/epoch_401_model.pth'))
    model_crossentropy.cuda()

    model_binary = Fasttext_model(300,ft_model,"False")
    model_binary.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_fasttext/experiments/exp_02_proper_split_time_17_34_13_date_11_08_2020/checkpoints/epoch_401_model.pth'))
    model_binary.cuda()

    model_mdn = MixtureDensityNetwork(300,3,4,ft_model,'normal')
    model_mdn.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_mdn/experiments/exp_07_correct_dataset_split_time_17_20_19_date_11_08_2020/checkpoints/epoch_401_model.pth'))
    model_mdn.cuda()


    train_predictions = torch.zeros((3,200,9*9*9))
    val_predictions = torch.zeros((3,50,9*9*9))

    _,train_targets = make_predictions(model_crossentropy,train_loader,"crossentropy",return_targets=True)
    _,val_targets = make_predictions(model_crossentropy,val_loader,"crossentropy",return_targets=True)


    counter = 0
    for model_name,model in zip(["crossentropy","binary","mdn"],[model_crossentropy,model_binary,model_mdn]):
        print(counter)
        train_predictions[counter] = make_predictions(model,train_loader,model_name)
        val_predictions[counter] = make_predictions(model,val_loader,model_name)
        counter += 1
        
    np.savez('data.npz', train_predictions=train_predictions, val_predictions=val_predictions,train_targets = train_targets,val_targets=val_targets)