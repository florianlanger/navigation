import torch
import pandas as pd
import csv
import numpy as np

class Sampler(torch.utils.data.Sampler):
    def __init__(self,config,path_to_dataset_csv,min_index,max_index):
        self.config = config
        self.pairs_per_epoch = config["training"]["pairs_per_epoch"]
        self.df = pd.read_csv(path_to_dataset_csv,quoting=csv.QUOTE_NONNUMERIC,header=None)
        self.min_index = min_index
        self.max_index = max_index

    def sample_index_outside_no_fly(self):
        while True:
            index = torch.randint(self.min_index,self.max_index,(1,)).cuda()
            position = torch.from_numpy(self.df.iloc[index.item(),1:4].to_numpy().astype(np.float64)).cuda()
            if not (torch.all(position >= self.config["sampler"]["no_fly_zone"][0,0]) and torch.all(position <= self.config["sampler"]["no_fly_zone"][0,1])):
                if not (torch.all(position >= self.config["sampler"]["no_fly_zone"][1,0]) and torch.all(position <= self.config["sampler"]["no_fly_zone"][1,1])):
                    break
        return index

    def sample_single_index(self):
        if self.config["sampler"]["no_fly"] == 'False':
            return torch.randint(self.min_index,self.max_index,(1,)).cuda()
        elif self.config["sampler"]["no_fly"] == 'True':
            return self.sample_index_outside_no_fly()

    
    def __iter__(self):
        self.counter = -1
        return self

    def __len__(self):
        return self.pairs_per_epoch

    def __next__(self):            
        if self.counter < self.pairs_per_epoch-1:
            self.counter += 1
            return self.sample_single_index()
        else:
            raise StopIteration 
