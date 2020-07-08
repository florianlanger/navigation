import torch
from torch.distributions.multivariate_normal import MultivariateNormal
def calc_loss(parameters,target_pose,cube):
    #covariance_matrix = torch.zeros((3,3))
    #covariance_matrix[0,0],covariance_matrix[1,1], covariance_matrix[2,2] = parameters[3],parameters[4],parameters[5]
    m = MultivariateNormal(parameters[:3],covariance_matrix=0.01 * torch.eye(3))
    loss = - m.log_prob(target_pose)
    # probability = torch.exp(-loss)
    # print(probability)
    return loss


def calc_dist(parameters,target_pose):
    return torch.norm((parameters[:3] - target_pose))
    
#calc_loss(torch.tensor([0.,1.,2.]),torch.tensor([0.2,1.,2.]),None)