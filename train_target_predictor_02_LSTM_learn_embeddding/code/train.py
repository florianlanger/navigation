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


from model import MixtureDensityNetwork
from loss import calc_loss_single_node,calc_loss_two_nodes,calc_correct,calc_Euclidean_distance
from visualisations import visualise,plot_history
from data import Target_Predictor_Dataset
from utilities import convert_pose_to_one_hot, create_directories,write_to_file, calc_node_probabilities


def train(model,optimizer,data_loader,config,epoch,exp_path,kind):
    total_loss = 0
    total_correct = 0
    for i,(descriptions,target_poses,vectorized_descriptions,descriptions_lengths) in enumerate(data_loader):

        model.zero_grad()
        pi,normal = model(vectorized_descriptions,descriptions_lengths)
        loss = model.loss(vectorized_descriptions,descriptions_lengths,target_poses).mean()
        
        
        if kind =='train':
            loss.backward()
            optimizer.step()

        #target = convert_pose_to_one_hot(target_poses)
        predicted_probabilities = calc_node_probabilities(pi, normal)

        with torch.no_grad():
            total_loss += loss.item()
            #total_correct += calc_correct(predicted_probabilities,target)

            if (epoch -1) % config["visualisations"]["interval"] ==0:
                if kind =='train':
                    if i == 0:
                        visualise(predicted_probabilities,target_poses.detach().cpu().numpy(),descriptions,'{}/visualisations/predictions/train/epoch_{}'.format(exp_path,epoch),config,pi,normal)

                if kind =='val':
                    if i == 0:
                        visualise(predicted_probabilities,target_poses.detach().cpu().numpy(),descriptions,'{}/visualisations/predictions/val/epoch_{}'.format(exp_path,epoch),config,pi,normal)

    average_loss = total_loss/(i+1)
    accuracy = total_correct/(float(i+1)*target_poses.shape[0])

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

    create_directories(exp_path)

    #create vocab
    descriptions_strings = []
    with open('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new.csv', 'r') as csv_file:
        for i in range(250):
            line = csv_file.readline()
            descriptions_strings.append(line.rsplit(',', 1)[1].rstrip("\n").replace('.','').lower())
    vocab = ['<pad>'] + sorted(set([word for sentence in descriptions_strings for word in sentence.split()]))
    
    # dataset = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new.csv',250)
    # train_data,val_data = torch.utils.data.random_split(dataset,(200,50))
    train_data = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_training.csv',200,vocab)
    val_data = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_validation.csv',50,vocab)
    #train_data,val_data = data.Subset(dataset,[0,1,2,3]),data.Subset(dataset, [4])
    train_loader = data.DataLoader(train_data, batch_size = config["training"]["batch_size"], shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size = config["training"]["batch_size"],shuffle=False)
       
    # load model and optimiser

    # print('loading fasttext model')
    # if config["mode"] == "debug":
    #     print("Debug mode!")
    #     ft_model = None
    # elif config["mode"] == "normal":
    #     ft_model = fasttext.load_model('/data/cvfs/fml35/original_downloads/Fasttext/cc.en.300.bin')
    # print('finished loading fasttext model')
    model = MixtureDensityNetwork(config["model"]['embedding_dim'],3,config["model"]["number_modes"],len(vocab))

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(1,config["training"]["n_epochs"]+1):
        train(model,optimizer,train_loader,config,epoch,exp_path,kind='train')
        train(model,optimizer,val_loader,config,epoch,exp_path,kind='val')
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)

main()



