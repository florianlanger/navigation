import torch
import torchvision
from torch.utils import data
import pandas as pd 
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import os.path
import csv
import shutil

from data import Sampler

class Fake_Dataset(data.Dataset):
  def __init__(self):
      pass

  def __getitem__(self, number):
      return number[0],number


def loss_batch(batch):
    return batch % 20

examples_per_epoch = 10
batch_size = 5
percentage_of_hard = 0.5

training_data = Fake_Dataset()
sampler = Sampler(66,examples_per_epoch,0.5)
train_loader =  torch.utils.data.DataLoader(training_data, batch_size = batch_size, sampler=sampler)

losses = torch.empty(examples_per_epoch)
indices = torch.empty(examples_per_epoch,2)
for epoch in range(4):
    print(train_loader.sampler.pairs)
    for batch_id,(data,index) in enumerate(train_loader):
        loss = loss_batch(data)
        losses[batch_id*batch_size:(batch_id+1)*batch_size] = loss
        indices[batch_id*batch_size:(batch_id+1)*batch_size] = index
    if epoch == 2:
        sampler.set_ratio_hard_pairs(0.3)
    sampler.update_sampler(losses, indices,'hard_pairs_test.txt')
