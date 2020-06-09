import torch
from torch.utils import data
from numpy import pi
import os.path
import networkx as nx
import sys

sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/code"))
sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/code/data"))

class Dataset_Target_Adjuster(data.Dataset):
    def __init__(self,corners_space,corners_no_fly_zone,config):
        'Initialization'
        self.config = config
        self.corners_space = corners_space
        self.corners_no_fly_zone = corners_no_fly_zone

    def __len__(self):
        return self.config["training"]["pairs_per_epoch"]

    def instruction_to_vector(self,theta,instruction):
        theta = theta * 2 * pi
        l = 0.4
        if instruction == 0:
            return torch.tensor([0.,0.,0.,0.,]).cuda()
        elif instruction == 1:
            return torch.tensor([-torch.sin(theta).item()*l,torch.cos(theta).item()*l,0.,0.,]).cuda()
        elif instruction == 2:
            return torch.tensor([-torch.sin(theta+pi).item()*l,torch.cos(theta+pi).item()*l,0.,0.,]).cuda()
        elif instruction == 3:
            return torch.tensor([-torch.sin(theta+pi/2).item()*l,torch.cos(theta+pi/2).item()*l,0.,0.,]).cuda()
        elif instruction == 4:
            return torch.tensor([-torch.sin(theta+3*pi/2).item()*l,torch.cos(theta+3*pi/2).item()*l,0.,0.,]).cuda()
        elif instruction == 5:
            return torch.tensor([0.,0.,l,0.,]).cuda()
        elif instruction == 6:
            return torch.tensor([0.,0.,-l,0.,]).cuda()
        elif instruction == 7:
            return torch.tensor([0.,0.,0.,0.25,]).cuda()
        elif instruction == 8:
            return torch.tensor([0.,0.,0.,-0.25,]).cuda()

    def sample_pose(self):
        while True:
            pose = self.corners_space[0] + torch.rand((4,)).cuda() * (self.corners_space[1]-self.corners_space[0])
            if self.check_flyable(pose):
                break
        return pose

    def check_flyable(self,pose):
        in_no_fly_1 = (torch.all(pose[:3] + 0.0001 >= self.corners_no_fly_zone[0,0]) and torch.all(pose[:3] - 0.0001<= self.corners_no_fly_zone[0,1]))
        in_no_fly_2 = (torch.all(pose[:3] + 0.0001 >= self.corners_no_fly_zone[1,0]) and torch.all(pose[:3] - 0.0001<= self.corners_no_fly_zone[1,1]))
        if (in_no_fly_1 or in_no_fly_2):
            return False
        else:
            return True

    def __getitem__(self, indices=None):
        pose = self.sample_pose()
        target_pose_valid = False
        while not target_pose_valid:
            instruction = torch.randint(9,(1,)).item()
            pose_diff_vector = self.instruction_to_vector(pose[3],instruction)
            target_pose = pose + pose_diff_vector
            target_pose[3] = target_pose[3] % 1.
            if self.check_flyable(target_pose):
                target_pose_valid = True
        one_hot_instruction = torch.zeros((9,)).cuda()
        one_hot_instruction[instruction] = 1.

        return pose, one_hot_instruction, pose_diff_vector




   