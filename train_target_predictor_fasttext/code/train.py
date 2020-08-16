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



from model import Fasttext_model
from loss import calc_loss_single_node,calc_loss_two_nodes,calc_correct
from visualisations import visualise,plot_history
from data import Target_Predictor_Dataset
from utilities import convert_pose_to_one_hot, create_directories,write_to_file


def train(model,optimizer,data_loader,config,epoch,exp_path,kind):
    total_loss = 0
    total_correct = 0
    for i,(cube_dimensions,descriptions,target_poses) in enumerate(data_loader):

        model.zero_grad()
        output = model(descriptions)

        if model.cross_entropy == "False":
            softmax = nn.Softmax(dim=2)
            output = softmax(output.reshape((-1,9*9*9,2)))
            target = convert_pose_to_one_hot(cube_dimensions,target_poses)
            loss = calc_loss_two_nodes(output, target)
            output_for_visualisation = output[:,:,0].detach().cpu()
            #loss = calc_loss_single_node(output, target)
        
        if model.cross_entropy == "True":
            softmax = nn.Softmax(dim=1)
            output = softmax(output.reshape((-1,9*9*9)))
            target = convert_pose_to_one_hot(cube_dimensions,target_poses)
            loss = calc_loss_single_node(output, target)
            output_for_visualisation = output.detach().cpu()
            #loss = calc_loss_single_node(output, target)
        
        if kind =='train':
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()

            if (epoch -1) % config["visualisations"]["interval"] ==0:
                if kind =='train':
                    if i == 0:
                        visualise(output_for_visualisation,target.detach().cpu(),np.array(cube_dimensions.detach().cpu()),descriptions,'{}/visualisations/predictions/train/epoch_{}'.format(exp_path,epoch),config)

                if kind =='val':
                    if i == 0:
                        visualise(output_for_visualisation,target.detach().cpu(),np.array(cube_dimensions.detach().cpu()),descriptions,'{}/visualisations/predictions/val/epoch_{}'.format(exp_path,epoch),config)

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
    print('after exp path')

    create_directories(exp_path)
    print('before data')
    train_data = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_training.csv',200)
    val_data = Target_Predictor_Dataset('/scratches/robot_2/fml35/mphil_project/navigation/target_pose/training_data1/data_transcribed_new_validation.csv',50)
    #train_data,val_data = torch.utils.data.random_split(dataset,(200,50))
    train_loader = data.DataLoader(train_data, batch_size = config["training"]["batch_size"], shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size = config["training"]["batch_size"],shuffle=False)
       
    # load model and optimiser
    print('loading fasttext model')
    ft_model = fasttext.load_model('/data/cvfs/fml35/original_downloads/Fasttext/cc.en.300.bin')
    print('finished loading fasttext model')
    model = Fasttext_model(config["model"]["embedding_dim"],ft_model,config["model"]["cross_entropy"])

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(1,config["training"]["n_epochs"]+1):
        train(model,optimizer,train_loader,config,epoch,exp_path,kind='train')
        train(model,optimizer,val_loader,config,epoch,exp_path,kind='val')
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)

main()



