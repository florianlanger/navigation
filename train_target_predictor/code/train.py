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

from model import LSTM_model
from loss import calc_loss_single_node,calc_loss_two_nodes,calc_correct
from visualisations import visualise,plot_history
from data import Target_Predictor_Dataset
from utilities import convert_pose_to_one_hot, create_directories,write_to_file


def train(model,optimizer,data_loader,config,epoch,exp_path,kind):
    total_loss = 0
    total_correct = 0
    for i,(cube_dimensions,descriptions,length_descriptions,text_descriptions,target_poses) in enumerate(data_loader):

        model.zero_grad()
        output = model(cube_dimensions,descriptions,length_descriptions)

        softmax = nn.Softmax(dim=2)
        output = softmax(output.reshape((-1,9*9*9,2)))

        target = convert_pose_to_one_hot(cube_dimensions,target_poses)
        loss = calc_loss_two_nodes(output, target)
        #loss = calc_loss_single_node(output, target)
        
        if kind =='train':
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_correct += calc_correct(output[:,:,0],target)

            if (epoch -1) % config["visualisations"]["interval"] ==0:
                if kind =='train':
                    if i == 0:
                        visualise(output[:,:,0].detach().cpu(),target.detach().cpu(),np.array(cube_dimensions.detach().cpu()),text_descriptions,'{}/visualisations/predictions/train/epoch_{}'.format(exp_path,epoch),config)

                if kind =='val':
                    if i == 0:
                        visualise(output[:,:,0].detach().cpu(),target.detach().cpu(),np.array(cube_dimensions.detach().cpu()),text_descriptions,'{}/visualisations/predictions/val/epoch_{}'.format(exp_path,epoch),config)

    average_loss = total_loss/(i+1)
    accuracy = total_correct/(float(i+1)*cube_dimensions.shape[0])

    print('Epoch: {} Average {} loss: {:.4f} Accuracy: {:.4f} \n'.format(epoch,kind,average_loss,accuracy))


    if kind == 'train':
        if epoch == 1:
            write_to_file(exp_path + "/history.csv",'epoch,train_loss,train_acc,val_loss,val_acc\n')
        if (epoch-1) % config["training"]["save_interval"] == 0:
            torch.save(model.state_dict(), exp_path + '/checkpoints/epoch_{}_model.pth'.format(epoch))
        torch.save(model.state_dict(), exp_path + '/checkpoints/last_epoch_model.pth')
        write_to_file(exp_path + '/history.csv',"{},{},{}".format(epoch,average_loss,accuracy))

    if kind == 'val':
        write_to_file(exp_path + '/history.csv',",{},{}\n".format(average_loss,accuracy))



def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load config
    with open('{}/../config.json'.format(dir_path), 'r') as file:
        config = json.load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    exp_path = '{}/../experiments/{}_{}'.format(dir_path,config["experiment_name"], datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    #create directories for checkpoints, visualisations and copy code and config
    create_directories(exp_path)

    dataset = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new.csv',250)
    train_data,val_data = torch.utils.data.random_split(dataset,(200,50))
    train_loader = data.DataLoader(train_data, batch_size = config["training"]["batch_size"], shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size = config["training"]["batch_size"],shuffle=True)
            
    # load model and optimiser

    model = LSTM_model(config["model"]["embedding_dim"],config["model"]["hidden_dim"],dataset.len_vocab)


    # for parameters in model.parameters():
    #     print(parameters.shape)
        #print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(1,config["training"]["n_epochs"]+1):
        train(model,optimizer,train_loader,config,epoch,exp_path,kind='train')
        train(model,optimizer,val_loader,config,epoch,exp_path,kind='val')
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)

main()



