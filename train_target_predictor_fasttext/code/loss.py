import torch

def calc_loss_single_node(output,target):
    return - torch.mean(target * torch.log(output))


def calc_loss_two_nodes(output,target):
    return - torch.mean(10 * target * torch.log(output[:,:,0]) + (1-target)* torch.log(output[:,:,1]))


def calc_correct(output,target):
    #print('output',output)
    #print('target',target)
    _,argmax_output =  torch.max(output,dim=1)
    
    number_correct = 0
    for i in range(output.shape[0]):
        if (((target[i] == 1.).nonzero().flatten() - argmax_output[i]) == 0).any().item() == True:
            number_correct += 1
    return number_correct
