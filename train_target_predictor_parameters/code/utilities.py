import os
import shutil
import numpy as np
import torch



def convert_pose_to_one_hot(cube,target_pose):
    print('cube',cube)
    print('target_pose',target_pose)
    cube_position = np.array(cube[:3])
    cube_size = np.array(cube[3:6])
    target_position = np.array(target_pose)

    indices = np.round((target_position - cube_position) / (cube_size/2))
    print(indices)
    indices += 4

    #-4, -4, -4 is 0 then go first in z dir, then y then x
    one_hot = torch.zeros(729)
    one_hot[int(indices[0] * 81 + indices[1]*9 + indices[2])] = 1
    print(indices)
    print(one_hot)
    return one_hot


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