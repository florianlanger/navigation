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

from model import LSTM_model
from loss import calc_loss,calc_dist
from visualisations import visualise,plot_history
from utilities import convert_pose_to_one_hot, create_directories,write_to_file


def train(model,optimizer,all_train_data,config,epoch,exp_path,kind):
    total_loss = 0
    total_dist = 0
    for i,training_dict in enumerate(all_train_data):

        model.zero_grad()

        description = training_dict['description']
        cube_dimensions = torch.tensor(training_dict['cube'])
        target_pose = torch.tensor(training_dict['target_pose'])

        parameters = model(description,cube_dimensions)

        loss = calc_loss(parameters,target_pose,cube_dimensions)
        
        if kind =='train':
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_dist += calc_dist(parameters,target_pose)

            # if (epoch -1) % config["visualisations"]["interval"] ==0:
            #     if kind =='train':
            #         if i < config["visualisations"]["number"]:
            #             visualise(output.detach(),target.detach(),np.array(cube_dimensions.detach()),training_dict['description'],'{}/visualisations/predictions/train/epoch_{}_example_{}.png'.format(exp_path,epoch,i))

            #     if kind =='val':
            #         if i < config["visualisations"]["number"]:
            #             visualise(output.detach(),target.detach(),np.array(cube_dimensions.detach()),training_dict['description'],'{}/visualisations/predictions/val/epoch_{}_example_{}.png'.format(exp_path,epoch,i))



    if kind == 'train':
        if epoch == 1:
            write_to_file(exp_path + "/history.csv",'epoch,train_loss,train_dist,val_loss,val_dist\n')
        if (epoch-1) % config["training"]["save_interval"] == 0:
            torch.save(model.state_dict(), exp_path + '/checkpoints/epoch_{}_model.pth'.format(epoch))
        torch.save(model.state_dict(), exp_path + '/checkpoints/last_epoch_model.pth')
        write_to_file(exp_path + '/history.csv',"{},{},{}".format(epoch,total_loss/(i+1),total_dist/float(i+1)))

    if kind == 'val':
        write_to_file(exp_path + '/history.csv',",{},{}\n".format(total_loss/(i+1),total_dist/float(i+1)))



def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load config
    with open('{}/../config.json'.format(dir_path), 'r') as file:
        config = json.load(file)

    exp_path = '{}/../experiments/{}_{}'.format(dir_path,config["experiment_name"], datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    #create directories for checkpoints, visualisations and copy code and config
    create_directories(exp_path)

    # load word to vector model
    with open('{}/../cached_word_to_vector.json'.format(dir_path), 'rb') as fp:
        wv = pickle.load(fp)

    # load train data
    train_data = []
    val_data = []
    with open('{}/../../target_pose/training_data/data.csv'.format(dir_path), 'r') as csv_file:
            for _ in range(80):
                line = csv_file.readline()
                train_data.append(json.loads(line))
            for _ in range(80,89):
                line = csv_file.readline()
                val_data.append(json.loads(line))


            
    # load model and optimiser

    model = LSTM_model(25,config["model"]["hidden_dim"],wv)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(1,config["training"]["n_epochs"]+1):
        train(model,optimizer,train_data,config,epoch,exp_path,kind='train')
        train(model,optimizer,val_data,config,epoch,exp_path,kind='val')
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)

main()



