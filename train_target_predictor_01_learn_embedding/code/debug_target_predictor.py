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
import sys

from model import LSTM_model
from loss import calc_loss_single_node,calc_loss_two_nodes,calc_correct
from visualisations import visualise,plot_history
from data import Target_Predictor_Dataset
from utilities import convert_pose_to_one_hot, create_directories,write_to_file


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
dataset = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new.csv',250)

lstm_model = LSTM_model(5,32,63)

lstm_model.load_state_dict(torch.load('/scratches/robot_2/fml35/mphil_project/navigation/train_target_predictor/experiments/debug_testing_time_14_44_52_date_15_07_2020/checkpoints/epoch_161_model.pth'))
lstm_model.cuda()
lstm_model.eval()
text = 'to the right of the cube but further back on central height'
split_text = text.split()
j = 0
vectorized_description = torch.zeros((1,40),dtype=torch.int64).cuda()
length = torch.tensor([12],dtype=torch.int64).cuda()
while j < len(split_text):
    vectorized_description[0,j] = dataset.vocab.index(split_text[j])
    j += 1

output = lstm_model(None,vectorized_description,length)
softmax = nn.Softmax(dim=1)
output = softmax(output.reshape((9*9*9,2)))
np.set_printoptions(threshold=sys.maxsize)
print(output[:,0].detach().cpu().numpy().round(1))

_,index = torch.max(output[:,0].flatten(),dim=0)
index = index.item()
print('index',index)
indices = np.array([index // 81,  (index % 81)  // 9, index % 9])
print('####################')
print('indices', indices)
