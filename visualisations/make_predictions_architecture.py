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

from train_target_predictor_01_learn_embedding.code.data import Target_Predictor_Dataset as data_01
from train_target_predictor_01_learn_embedding.code.model import MixtureDensityNetwork as model_01

from train_target_predictor_02_LSTM_learn_embeddding.code.data import Target_Predictor_Dataset as data_02
from train_target_predictor_02_LSTM_learn_embeddding.code.model import MixtureDensityNetwork as model_02

from train_target_predictor_03_mdn.code.data import Target_Predictor_Dataset as data_03
from train_target_predictor_03_mdn.code.model import MixtureDensityNetwork as model_03

from train_target_predictor_04_fasttext_and_lstm.code.data import Target_Predictor_Dataset as data_04
from train_target_predictor_04_fasttext_and_lstm.code.model import MixtureDensityNetwork as model_04

from train_target_predictor_03_mdn.code.utilities import calc_node_probabilities


def make_predictions(model,data_loader,bs):

    outputs = torch.zeros((len(data_loader.dataset),9*9*9))

    for i,(descriptions,target_poses,vectorized_descriptions,descriptions_lengths) in enumerate(data_loader):

        pi,normal = model(vectorized_descriptions)#,descriptions_lengths)
        output_for_visualisation = calc_node_probabilities(pi, normal)
        
        outputs[i*bs:(i+1)*bs,:] = output_for_visualisation

    return outputs



def save_predictions():
    bs = 25

    descriptions_strings = []
    with open('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new.csv', 'r') as csv_file:
        for i in range(250):
            line = csv_file.readline()
            descriptions_strings.append(line.rsplit(',', 1)[1].rstrip("\n").replace('.','').lower())
    vocab = ['<pad>'] + sorted(set([word for sentence in descriptions_strings for word in sentence.split()]))


    train_data = data_01('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_training.csv',200,vocab)
    val_data = data_01('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_validation.csv',50,vocab)
    train_loader = data.DataLoader(train_data, batch_size = bs, shuffle=False)
    val_loader = data.DataLoader(val_data, batch_size = bs,shuffle=False)
       


    #ft_model = fasttext.load_model('/data/cvfs/fml35/original_downloads/Fasttext/cc.en.300.bin')
    model = model_01(5,3,4,len(vocab))
    
    model.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_01_learn_embedding/experiments/exp_01_time_16_46_17_date_12_08_2020/checkpoints/epoch_201_model.pth'))
    #model.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_02_LSTM_learn_embeddding/experiments/exp_01_time_17_12_03_date_12_08_2020/checkpoints/epoch_101_model.pth'))
    #model.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_03_mdn/experiments/exp_07_correct_dataset_split_time_17_20_19_date_11_08_2020/checkpoints/epoch_401_model.pth'))
    #model.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor_04_fasttext_and_lstm/experiments/exp_01_time_17_44_05_date_12_08_2020/checkpoints/epoch_81_model.pth'))

    model.cuda()



    train_predictions = make_predictions(model,train_loader,bs)
    val_predictions = make_predictions(model,val_loader,bs)

        
    np.savez('data_architectures/architecture_01.npz', train_predictions=train_predictions, val_predictions=val_predictions)

save_predictions()