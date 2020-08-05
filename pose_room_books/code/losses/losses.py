import torch
from scipy.spatial.transform import Rotation as R
import numpy as np


def pose_losses(outputs,targets):
    return torch.sqrt(torch.sum((outputs - targets)**2,dim=1))

def L2_distances(outputs,targets):
    return torch.sqrt(torch.sum((outputs[:,:3] - targets[:,:3])**2,dim=1))

def angle_differences(outputs,targets):
    angle_difference = torch.zeros(outputs.shape[0]).cuda()
    for i in range(outputs.shape[0]):
        r1 = R.from_quat(outputs[i,3:7].cpu().numpy().reshape(-1))
        r2 = R.from_quat(targets[i,3:7].cpu().numpy().reshape(-1))
        r3 = r2 * r1.inv()
        angle_difference[i] = np.linalg.norm(r3.as_rotvec()) * 180 / np.pi
    return angle_difference


# outputs = torch.zeros((1,7))
# targets = torch.zeros((1,7))

# outputs[0,3:7] = torch.tensor(list(R.from_euler('xyz',[90,90,90],degrees=True).as_quat()))
# targets[0,3:7] = torch.tensor(list(R.from_euler('xyz',[0,0,0],degrees=True).as_quat()))

# print(angle_difference(outputs,targets))