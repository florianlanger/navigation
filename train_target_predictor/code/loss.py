import torch

def calc_loss(output,target):
    return - torch.mean(10 * target * torch.log(output) + (1-target)* torch.log(1-output))


def loss(output,target):
    output = torch.reshape(output,(-1,9*9*9,2))
    torch.exp(output[target]) / torch.sum(torch.exp(output),dim=0)

def calc_correct(output,target):
    #print('output',output)
    #print('target',target)
    _,argmax_output =  torch.max(output,dim=1)
    number_correct = 0
    for i in range(output.shape[0]):
        if (((target[i] == 1.).nonzero().flatten() - argmax_output[i]) == 0).any().item() == True:
            number_correct += 1
    return number_correct
