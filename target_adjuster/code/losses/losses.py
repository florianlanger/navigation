import torch

def pose_loss(outputs,targets):
    position_loss = L2_distance(outputs[:,:3],targets[:,:3])
    angle_diff = angle_difference(outputs[:,3],targets[:,3])
    return position_loss + 3 * angle_diff

def L2_distance(outputs,targets):
    return torch.mean(torch.sqrt(torch.sum((outputs - targets)**2,dim=1)))

def angle_difference(outputs,targets):
    abs_angle_diff = torch.abs(outputs-targets)%1.
    abs_angle_diff[abs_angle_diff>0.5] = 1. - abs_angle_diff[abs_angle_diff>0.5]
    return torch.mean(abs_angle_diff)