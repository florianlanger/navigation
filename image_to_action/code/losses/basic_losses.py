import torch.nn as nn
import torch


def classify_predictions(output,action_penalties):
    #classify each example in batch as correct or wrong
    _ , indices = torch.max(output,dim=1)
    return action_penalties.gather(1, indices.view(-1,1))<0.00001

 
# def calc_action_penalties_hard_coded(positions, terminate_penalty):
#     batch_size = positions.shape[0]
#     #Have to set correct stepsize for each dimension
#     dx,dy,dz = 0.1, 0.1, 0.1
#     # action0 does nothing, action 1 moves in positive x direction, 2 in negative x, 3 in postive y, 4 in negative y, 5 positive z, 6 negative z, 7 terminate
#     p0 = torch.abs(positions[:,0] - positions[:,3]) + torch.abs(positions[:,1] - positions[:,4]) + torch.abs(positions[:,2] - positions[:,5])
#     p1 = torch.abs(positions[:,0] + dx - positions[:,3]) + torch.abs(positions[:,1] - positions[:,4]) + torch.abs(positions[:,2] - positions[:,5])
#     p2 = torch.abs(positions[:,0] - dx - positions[:,3]) + torch.abs(positions[:,1] - positions[:,4]) + torch.abs(positions[:,2] - positions[:,5])
#     p3 = torch.abs(positions[:,0] - positions[:,3]) + torch.abs(positions[:,1] + dy - positions[:,4]) + torch.abs(positions[:,2] - positions[:,5])
#     p4 = torch.abs(positions[:,0] - positions[:,3]) + torch.abs(positions[:,1] - dy - positions[:,4]) + torch.abs(positions[:,2] - positions[:,5])
#     p5 = torch.abs(positions[:,0] - positions[:,3]) + torch.abs(positions[:,1] - positions[:,4]) + torch.abs(positions[:,2] + dz - positions[:,5])
#     p6 = torch.abs(positions[:,0] - positions[:,3]) + torch.abs(positions[:,1] - positions[:,4]) + torch.abs(positions[:,2] - dz - positions[:,5]) 
#     p = torch.stack((p0,p1,p2,p3,p4,p5,p6),dim=1)
#     min_p,_ = torch.min(p,dim=1)
#     action_penalties = p-min_p.view(-1,1).double()
#     p7 = torch.ones((batch_size,1)).cuda().double() * terminate_penalty
#     for i in range(batch_size):
#         if torch.norm(positions[i,:3]-positions[i,3:6]) < 0.00001:
#             p7[i,0] = 0
#             # Ensure that when at goal move away have loss 0.2, and when stay 
#             action_penalties[i] *= 2
#             action_penalties[i,0] = 0.1
#     action_penalties = torch.cat((action_penalties,p7), dim=1)
#     return action_penalties




############## compute final loss as dot product or with KL-div ######################

def calc_individual_loss_dot_product(output,action_penalties):
    number_outputs = action_penalties.shape[1]
    weighted_loss = torch.bmm(output.view(-1, 1, number_outputs).double(), action_penalties.view(-1, number_outputs, 1))
    return weighted_loss

def calc_individual_loss_kl_div(output,action_penalties):
    target_dist_batch = calc_target_distribution(action_penalties)
    log_output = torch.log(output + 0.00001)
    return torch.sum(torch.nn.functional.kl_div(log_output.double(),target_dist_batch, reduction='none'),dim=1)

def calc_target_distribution(action_penalties):
    action_penalties_sum = torch.sum(action_penalties < 0.0001,dim=1)
    assert torch.min(action_penalties_sum).cpu() > 0.0001, "Some of the action penalties are too small"
    target_dist_batch = ((action_penalties < 0.0001).to(torch.float16)/ action_penalties_sum.view(-1,1).to(torch.float64)).double()
    return target_dist_batch



################## Pointer functions ########################


def calc_individual_loss(output,action_penalties,config):
    if config["loss"]["type"] == "dot_product":
        return calc_individual_loss_dot_product(output,action_penalties)
    elif config["loss"]["type"] == "kl_div":
        return calc_individual_loss_kl_div(output,action_penalties)


# output = torch.tensor([[0.1,0.3,0.1,0.2,0.1,0.1,0.2,0.1],
# [0.5,0.1,0.1,0.2,0.1,0.1,0.1,0.1]])

# action_penalties = torch.tensor([[0.1,0.3,0.0,0.2,0.0,0.1,0.2,0.1],
# [0.5,0.0,0.1,0.0,0.1,0.1,0.0,0.1]])


# calc_individual_loss_kl_div(output,action_penalties)