import os
import shutil
import numpy as np
import torch



def convert_pose_to_one_hot(cube_dimensions,target_poses):
    indices = torch.round((target_poses - cube_dimensions[:,:3])/ (cube_dimensions[:,3:6]/2))

    indices += 4
    #-4, -4, -4 is 0 then go first in z dir, then y then x
    one_hot = torch.zeros(cube_dimensions.shape[0],729).cuda()

    neighbouring_indices = torch.tensor([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]).cuda()
    for i in range(cube_dimensions.shape[0]):
        # put 1 if just want single target, 7 if want all neighbouring
        for j in range(7):
            new_indices = indices[i] + neighbouring_indices[j]
            new_indices = torch.clamp(new_indices,0,8)
            one_hot[i,int(new_indices[0] * 81 + new_indices[1]*9 + new_indices[2])] = 1
        
    return one_hot

def calc_node_probabilities(pi, normal, cube_dimensions):
    node_probabilities = torch.zeros((cube_dimensions.shape[0],729)).cuda()
    for i in range(729):
        indices = torch.tensor([i //81, (i // 9) % 9, i % 9]).cuda()
        indices -= 4
        positions = cube_dimensions[:,:3] + indices * (cube_dimensions[:,3:6] /2.)
        
        positions = positions.unsqueeze(1)

        # print('normal mean',normal.mean)
        # print('normal stdev', normal.stddev)
        # print('positions',positions)
        # sum over contribution log contirbution from each of three dimensions x,y,z and exponentiate, shape is now [batchsize x number components]
        probabilities = torch.exp(torch.sum(normal.log_prob(positions),dim=2))
        
        # print('probs',probabilities)
        # sum over probabilities from each mode
        node_probabilities[:,i] = torch.sum(pi.probs * probabilities,dim=1)

    return node_probabilities.detach().cpu()


def create_directories(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path + '/visualisations')
    os.mkdir(exp_path + '/visualisations/predictions')
    os.mkdir(exp_path + '/visualisations/predictions/train')
    os.mkdir(exp_path + '/visualisations/predictions/val')
    os.mkdir(exp_path + '/visualisations/history')
    os.mkdir(exp_path + '/checkpoints')
    shutil.copy(exp_path +'/../../config.json',exp_path +'/config.json')
    shutil.copytree(exp_path +'/../../code',exp_path +'/code')

def write_to_file(path,content):
    log_file = open(path,'a')
    log_file.write(content)
    log_file.flush()
    log_file.close()