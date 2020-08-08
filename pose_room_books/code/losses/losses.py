import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def L2_distances(outputs,targets):
    return torch.sqrt(torch.sum((outputs - targets)**2,dim=1))

def angle_differences(outputs,targets):
    abs_angle_diff = torch.abs(outputs-targets)%1.
    abs_angle_diff[abs_angle_diff>0.5] = 1. - abs_angle_diff[abs_angle_diff>0.5]
    return abs_angle_diff

def pose_losses(outputs,targets):
    position_loss = L2_distances(outputs[:,:3],targets[:,:3])
    angle_diff = angle_differences(outputs[:,3],targets[:,3])
    return position_loss + angle_diff



#print(angle_differences(torch.tensor([1.1,0.5]),torch.tensor([0.2,0.7])))
# outputs = torch.zeros((1,7))
# targets = torch.zeros((1,7))

# outputs[0,3:7] = torch.tensor(list(R.from_euler('xyz',[90,90,90],degrees=True).as_quat()))
# targets[0,3:7] = torch.tensor(list(R.from_euler('xyz',[0,0,0],degrees=True).as_quat()))

# print(angle_difference(outputs,targets))