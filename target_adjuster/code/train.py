import torch
import torch.optim as optim
import torch.nn as nn
import json
import os
import numpy as np
import shutil
from datetime import datetime
import os.path
import sys

#sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation"))

from losses.losses import pose_loss,L2_distance,angle_difference
from data.dataset import Dataset_Target_Adjuster
from utilities import load_config,create_directories, write_to_file
from visualisations.plots import plot_history, visualise_target_predictions
from models.model import Target_Adjuster


def train(network, config, train_loader, optimizer, epoch, exp_path):
    network.train()
    total_loss,total_L2_dist,total_angle_diff = 0.,0.,0.
    for batch_idx, (poses,instructions,target_pose_diff_vectors) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = network(torch.cat((poses,instructions),dim=1))
        target_poses = poses + target_pose_diff_vectors
        predicted_poses = poses + outputs
        loss = pose_loss(predicted_poses,target_poses)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            L2_dist,angle_diff = L2_distance(predicted_poses[:,:3],target_poses[:,:3]),angle_difference(predicted_poses[:,3],target_poses[:,3])
            total_loss += loss
            total_L2_dist += L2_dist
            total_angle_diff += angle_diff
            if batch_idx % config["training"]["log_interval"] ==0:
                line = 'Epoch: {} Average loss: {:.4f} L2 dist: {:.4f} Angle Difference: {:.4f}'.format(
                            epoch,total_loss/((batch_idx +1)),total_L2_dist/((batch_idx +1)),total_angle_diff * 360 /((batch_idx +1)))
                print(line, end="\r", flush=True)
            if (epoch-1) % config["visualisations"]["interval"] == 0 and batch_idx == 0:
                visualise_target_predictions(poses.cpu(),instructions.cpu(),target_poses.cpu(),predicted_poses.cpu(),exp_path, config,epoch)
    print('\n')
    # This is only precise if have full number of batches per dataset
    total_loss /= (batch_idx + 1)
    total_L2_dist /= (batch_idx + 1)
    total_angle_diff /= (batch_idx + 1)


    if epoch == 1:
        write_to_file(exp_path + "/history.csv",'epoch,train_loss,train_L2_dist,train_angle_difference\n')
    if (epoch-1) % config["training"]["save_interval"] == 0:
        torch.save(network.state_dict(), exp_path + '/checkpoints/epoch_{}_model.pth'.format(epoch))
        torch.save(train_loader, exp_path + '/checkpoints/epoch_{}_data_loader.pth'.format(epoch))
    torch.save(network.state_dict(), exp_path + '/checkpoints/last_epoch_model.pth')
    torch.save(train_loader, exp_path + '/checkpoints/last_epoch_data_loader.pth')
    write_to_file(exp_path + '/history.csv',"{},{},{},{}\n".format(epoch,total_loss,total_L2_dist,total_angle_diff*360.))
    



def main():
    # get absolute path to current file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #os.environ["CUDA_VISIBLE_DEVICES"] = json.load(open('{}/../config.json'.format(dir_path),'r'))["gpu"]
    # load template config
    config = load_config('{}/../config.json'.format(dir_path))
    exp_path = '{}/../experiments/{}_{}'.format(dir_path,config["experiment_name"], datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    #create directories for checkpoints, visualisations and copy code and config
    create_directories(exp_path)
    # load dataset
    corners_space = torch.tensor([[-1.3,-0.5,0.2,0.],[1.8,1.4,1.7,1.]]).cuda()
    corners_no_fly_zone = torch.tensor([[[0.5,-0.5,0.1],[1.7,1.1,0.9]],[[-1.3,0.5,0.1],[-0.1,1.7,1.1]]]).cuda()
    dataset = Dataset_Target_Adjuster(corners_space,corners_no_fly_zone,config)

    # initialise model and optimiser
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    network = Target_Adjuster()
    network = nn.DataParallel(network)
    network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=config["training"]["learning_rate"])

    train_loader =  torch.utils.data.DataLoader(dataset, batch_size = config["training"]["batch_size"])

    epochs = range(1,config["training"]["n_epochs"]+1)
    for epoch in epochs:
        train(network,config,train_loader,optimizer,epoch, exp_path)
        if (epoch-1) % config["visualisations"]["interval"] == 0:
            plot_history('{}/history.csv'.format(exp_path),'{}/visualisations/history'.format(exp_path),epoch)

if __name__ == "__main__":
    main()