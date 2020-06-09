import torch.nn as nn
import torch


def classify_predictions(output,target_distributions):
    #classify each example in batch as correct or wrong
    _ , indices = torch.max(output,dim=1)
    return target_distributions.gather(1, indices.view(-1,1))>0.00001

############## compute final loss as dot product or with KL-div ######################

def calc_individual_loss(output,target_dist_batch):
    log_output = torch.log(output + 0.00001)
    return torch.sum(torch.nn.functional.kl_div(log_output.double(),target_dist_batch, reduction='none'),dim=1)

