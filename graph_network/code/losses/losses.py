import torch

def classify_predictions(predictions,targets):
    #classify each example in batch as correct or wrong
    return (torch.abs(predictions-targets) < 0.05).view(-1)

def calc_individual_loss(predictions,targets):
    return ((predictions-targets)**2).view(-1)

