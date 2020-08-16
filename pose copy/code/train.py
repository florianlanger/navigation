import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import shutil
from absl import flags
from absl import app
from tqdm import tqdm
from datetime import datetime
from torchvision.models import resnet18
from torchvision.transforms import ColorJitter
import torch.nn as nn

from losses.losses import pose_loss,L2_distance,angle_difference
from data.sampler import Sampler
from models.model import Pose_Model
from data.dataset import Pose_Dataset
from utilities import load_config, create_directories, write_to_file
from visualisations.plots import visualise_poses, plot_history


def one_epoch(network, config, data_loader, optimizer, epoch, exp_path,kind):
    if kind == 'train':
        network.train()
    elif kind == 'val':
        network.eval()
    total_loss,total_L2_dist,total_angle_diff = 0.,0.,0.
    for batch_idx, (images,positions,occluded) in enumerate(data_loader):
        print('images shape',images.shape)
        optimizer.zero_grad()
        outputs = network(images)
        loss = pose_loss(outputs,positions)
        if kind == 'train':
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            L2_dist,angle_diff = L2_distance(outputs[:,:3],positions[:,:3]),angle_difference(outputs[:,3],positions[:,3])
            total_loss += loss
            total_L2_dist += L2_dist
            total_angle_diff += angle_diff
            if batch_idx % config["training"]["log_interval"] ==0:
                line = 'Epoch: {} Average {} loss: {:.4f} L2 dist: {:.4f} Angle Difference: {:.4f}'.format(
                            epoch,kind,total_loss/((batch_idx +1)),total_L2_dist/((batch_idx +1)),total_angle_diff * 360 /((batch_idx +1)))
                print(line, end="\r", flush=True)

            visualise_poses(batch_idx, epoch,images,positions,outputs,config, exp_path, kind)
    print('\n')
    # This is only precise if have full number of batches per dataset
    total_loss /= (batch_idx + 1)
    total_L2_dist /= (batch_idx + 1)
    total_angle_diff /= (batch_idx + 1)

    if kind == 'train':
        if epoch == 1:
            write_to_file(exp_path + "/history.csv",'epoch,train_loss,train_L2_dist,train_angle_difference,val_loss,val_L2_dist,val_angle_difference,L2_dist_weight,angle_difference_weight\n')
        if (epoch-1) % config["training"]["save_interval"] == 0:
            torch.save(network.state_dict(), exp_path + '/checkpoints/epoch_{}_model.pth'.format(epoch))
            torch.save(data_loader, exp_path + '/checkpoints/epoch_{}_data_loader.pth'.format(epoch))
        torch.save(network.state_dict(), exp_path + '/checkpoints/last_epoch_model.pth')
        torch.save(data_loader, exp_path + '/checkpoints/last_epoch_data_loader.pth')
        write_to_file(exp_path + '/history.csv',"{},{},{},{}".format(epoch,total_loss,total_L2_dist,total_angle_diff*360.))
    
    if kind == 'val':
        write_to_file(exp_path + '/history.csv',",{},{},{}\n".format(total_loss,total_L2_dist,total_angle_diff*360.))

def main():
    # get absolute path to current file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.environ["CUDA_VISIBLE_DEVICES"] = json.load(open('{}/../config.json'.format(dir_path),'r'))["gpu"]
    # load template config
    config = load_config('{}/../config.json'.format(dir_path))
    exp_path = '{}/../experiments/{}_{}'.format(dir_path,config["experiment_name"], datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    #create directories for checkpoints, visualisations and copy code and config
    create_directories(exp_path)
    # load dataset
    transform = ColorJitter(0.5,0.5,0.5,0.1)
    train_dataset = Pose_Dataset('/data/cvfs/fml35/own_datasets/grid_world/ignas_living_room_random_poses/train','positions.csv',config,transform=transform)
    val_dataset = Pose_Dataset('/data/cvfs/fml35/own_datasets/grid_world/ignas_living_room_random_poses/val','positions.csv',config,transform=transform)
    # initialise model and optimiser
    pretrained_model = resnet18(pretrained=True)
    pretrained_model.fc = nn.Sequential()
    network = Pose_Model(pretrained_model,512)
    network.cuda()
    
    optimizer = optim.Adam([{'params': network.parameters()}], lr=config["training"]["learning_rate"])
    
    train_sampler = Sampler(config,'/data/cvfs/fml35/own_datasets/grid_world/ignas_living_room_random_poses/train/positions.csv',0,300000)
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size = config["training"]["batch_size"], sampler=train_sampler)
    val_sampler = Sampler(config,'/data/cvfs/fml35/own_datasets/grid_world/ignas_living_room_random_poses/val/positions.csv',0,50000)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size = config["training"]["batch_size"], sampler=val_sampler)

    epochs = range(1,config["training"]["n_epochs"]+1)
    for epoch in epochs:
        one_epoch(network,config,train_loader,optimizer,epoch, exp_path,'train')
        one_epoch(network,config,val_loader,optimizer,epoch, exp_path,'val')
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)

if __name__ == "__main__":
    main()