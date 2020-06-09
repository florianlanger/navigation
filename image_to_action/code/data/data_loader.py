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
import random
import networkx as nx
import sys

sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/code"))
sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/code/data"))

from plots import show_images
from utilities import load_config
from data.action_penalties import calc_action_penalties, calc_action_penalties_hard_coded, calc_action_penalties_occluded
from graphs.conversions import Converter

class Dataset(data.Dataset):
      def __init__(self, dir_path, file_name, config, transform=None, kind="training"):
            'Initialization'
            self.dir_path = dir_path
            self.df = pd.read_csv(dir_path+'/'+file_name,quoting=csv.QUOTE_NONNUMERIC,header=None)
            self.number_images = len(self.df)
            self.config = config
            self.transform = transform
            self.kind = kind
            if config["graph"]["type"] == "all_adjacent" and config["data"]["place"] == 'department':
                  graph_path = os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/rotation.gpickle'
            elif config["graph"]["type"] == "only_forward" and config["data"]["place"] == 'department':
                  graph_path = os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/forward.gpickle'
            elif config["graph"]["type"] == "only_forward" and config["data"]["place"] == 'living_room':
                  if config["graph"]["no_fly"] == False:
                        graph_path = os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/forward_living_room.gpickle'
                  elif config["graph"]["no_fly"] == True:
                        graph_path = os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/no_fly_living_room.gpickle'
            self.graph = nx.read_gpickle(graph_path)

            if "occlusion_probability" in self.config["data"]:
                  self.occluder_tranforms = torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                    std=[1., 1.,1.])
                  if self.kind == "training":
                        self.occluder_dir = '/data/cvfs/fml35/derivative_datasets/coco/train2017/images/'
                        self.number_occluders = 87445
                  elif self.kind == "validation":
                        self.occluder_dir = '/data/cvfs/fml35/derivative_datasets/coco/val2017/images/'
                        self.number_occluders = 5000

      def __len__(self):
        return self.config["training"]["pairs_per_epoch"]

      def __getitem__(self, indices=None):

            def occlud(self,image):
                  check_not_gray = False
                  while check_not_gray == False:
                        rand_occluder = torch.randint(0,self.number_occluders,(1,))
                        occluder_name = '{}.jpg'.format(str(rand_occluder.item()).zfill(6))
                        occluder = Image.open(self.occluder_dir + occluder_name)
                        if occluder.mode == "RGB":
                              check_not_gray = True

                  size_occluder =  torch.randint(int(self.config["data"]["min_occlusion"] * image.shape[1]),int(self.config["data"]["max_occlusion"] * image.shape[1]),(2,))
                  x = torch.randint(0,image.shape[1] - size_occluder[0].item(),(1,))
                  y = torch.randint(0,image.shape[1] - size_occluder[1].item(),(1,))

                  random_crop = torchvision.transforms.RandomResizedCrop((size_occluder[0].item(),size_occluder[1].item()),ratio=(1.,1.))
                  cropped_occluder = random_crop(occluder)
                  normalised_occluder = self.occluder_tranforms(F.to_tensor(cropped_occluder))
                  image[:,x : x + size_occluder[0].item(), y : y + size_occluder[1].item()] = normalised_occluder
                  return image

            if indices is not None:
                  if indices.shape[0] != 2 or indices.dtype != torch.int64:
                        print(indices.dtype)
                        raise Exception("Two indices of type 'torch.int64' have to be specified")
            else:
                  indices = torch.randint(self.number_images, (2,)).cuda()
            image_1_name, image_2_name = self.df.iloc[indices[0].item(),0], self.df.iloc[indices[1].item(),0]
            # positions: x1,y1,z1,x2,y2,z2
            positions_1 = torch.from_numpy(self.df.iloc[indices[0].item(),1:].to_numpy().astype(np.float64))
            positions_2  = torch.from_numpy(self.df.iloc[indices[1].item(),1:].to_numpy().astype(np.float64))
            positions = torch.cat((positions_1,positions_2)).cuda()
            image1 = Image.open(self.dir_path +'/images/' + image_1_name)
            image2 = Image.open(self.dir_path +'/images/' + image_2_name)
            image1, image2  = F.to_tensor(image1)[:3,:,:].cuda(),F.to_tensor(image2)[:3,:,:].cuda()
            if self.transform:
                  image1 = self.transform(image1)
                  image2 = self.transform(image2)
            if "occlusion_probability" in self.config["data"]:
                  if torch.rand(1).item() <  self.config["data"]["occlusion_probability"]:
                        image1 = occlud(self,image1)
                        action_penalties = calc_action_penalties_occluded(self.config)
                  else:
                        action_penalties = calc_action_penalties(positions,indices,self.graph,self.config).cuda()
            else:
                  action_penalties = calc_action_penalties(positions,indices,self.graph,self.config).cuda()
            images = torch.stack((image1,image2))
            return images, positions, indices, action_penalties


class Sampler(data.Sampler):
      def __init__(self,config):
            self.config = config
            self.pairs_per_epoch = config["training"]["pairs_per_epoch"]
            self.ratio_hard_pairs = config["sampler"]["ratio_hard_pairs"]
            self.number_hard_pairs = int(self.pairs_per_epoch * self.ratio_hard_pairs)
            self.ratio_terminate_pairs = config["sampler"]["ratio_terminate_pairs"]
            # Note ratio terminate pairs is defined w.r.t. whole data set 
            self.min_new_terminate_pairs = int(self.ratio_terminate_pairs * self.pairs_per_epoch)
            self.converter = Converter(config["data"]["min_pos"],config["data"]["max_pos"],config["data"]["steps"],config["data"]["number_images"],config["data"]["no_fly_zone"])
            self.random()

      def update_sampler(self,hard_pairs):
            hard_pairs = hard_pairs[:self.number_hard_pairs].long()
            random_pairs = self.sample_allowed_pairs(int(self.pairs_per_epoch-self.number_hard_pairs))
            if self.min_new_terminate_pairs != 0:
                  random_pairs[:self.min_new_terminate_pairs] = self.sample_allowed_pairs(self.min_new_terminate_pairs,terminate=True)
            pairs = torch.cat((random_pairs,hard_pairs))
            self.pairs = pairs[torch.randperm(self.pairs_per_epoch)]

            # print('Debug mode: Hard pairs are in the back:')
            # self.pairs = pairs
            # print(self.pairs)

      def random(self):
            if self.min_new_terminate_pairs == 0:
                  self.pairs = self.sample_allowed_pairs(self.pairs_per_epoch)
            else:
                  self.pairs = torch.empty((self.pairs_per_epoch,2),dtype=torch.int64).cuda()
                  self.pairs[:self.min_new_terminate_pairs] = self.sample_allowed_pairs(self.min_new_terminate_pairs,terminate=True)
                  self.pairs[self.min_new_terminate_pairs:] = self.sample_allowed_pairs(self.pairs_per_epoch - self.min_new_terminate_pairs)
                  self.pairs = self.pairs[torch.randperm(self.pairs_per_epoch)]
                  
      def sample_allowed_pairs(self,number, terminate = False):
            if self.config["graph"]["no_fly"] == False and terminate == False:
                  return torch.randint(self.config["data"]["number_images"],(number,2)).cuda()
            elif self.config["graph"]["no_fly"] == False and terminate == True:
                  return torch.randint(self.config["data"]["number_images"],(number,1)).repeat(1,2).cuda()

            elif self.config["graph"]["no_fly"] == True:
                  pairs = torch.empty(number,2,dtype=torch.int64).cuda()
                  for i in range(number):
                        if terminate == False:
                              for j in range(2):
                                    pairs[i,j] = self.sample_index_outside_no_fly()
                        if terminate == True:
                              pairs[i] = self.sample_index_outside_no_fly().repeat(2)
                  return pairs

      def sample_index_outside_no_fly(self):
            while True:
                  index = torch.randint(self.config["data"]["number_images"],(1,)).cuda()
                  if self.converter.check_flyable_index(index):
                        break
            return index
      
      def __iter__(self):
          self.counter = -1
          return self

      def __len__(self):
        return self.pairs_per_epoch

      def __next__(self):            
            if self.counter < self.pairs_per_epoch-1:
                  self.counter += 1
                  return self.pairs[self.counter]
            else:
                  raise StopIteration 


# config = load_config('/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/config.json')
# sampler = Sampler(config)
# print(sampler.sample_allowed_pairs(20))
# normalise = torchvision.transforms.Normalize(mean=[0.4381, 0.4192, 0.3965],std=[0.0929, 0.0905, 0.0902])
# training_data = Dataset('/data/cvfs/fml35/own_datasets/grid_world/3d_cube_equidistance','positions.csv',config, transform = normalise)

# train_loader = torch.utils.data.DataLoader(training_data, batch_size = 5, sampler = sampler )
# for data in train_loader:
#       pass

# sampler = Sampler(2400,12)
# a = len(sampler)
# iterator = iter(sampler)
# for i in range(a):
#       print(next(iterator))




def load_data_set(config):
      if config["data"]["place"] == 'department':
            if config["data"]["orientation"] == "False":
                  normalise = torchvision.transforms.Normalize(mean=[0.4381, 0.4192, 0.3965],
                                 std=[0.0929, 0.0905, 0.0902])
                  dataset = Dataset('/data/cvfs/fml35/own_datasets/grid_world/3d_cube_one_direction','positions.csv',
                  config, transform = normalise)
            elif config["data"]["orientation"] == "True":
                  normalise = torchvision.transforms.Normalize(mean=[0.4404, 0.4192, 0.3936],
                                 std=[0.0904, 0.0861, 0.0844])
                  dataset = Dataset('/data/cvfs/fml35/own_datasets/grid_world/3d_cube_equidistance','positions.csv',
                  config, transform = normalise)
      if config["data"]["place"] == 'living_room':
            #normalise = torchvision.transforms.Normalize(mean=[0.3802, 0.3665, 0.3556],std=[0.1590, 0.1600, 0.1613])
            dataset = Dataset('/data/cvfs/fml35/own_datasets/grid_world/ignas_living_room','positions.csv',
                  config) #, transform = normalise)
      

      return dataset



def process_data():
      path = '/data/cvfs/fml35/own_datasets/grid_world/3d_cube_equidistance/'
      df = pd.read_csv(path + 'images/positions.csv', header=None)
      kinds = ['training','validation','testing']
      numbers = [8000,1000,1000]
      for kind,number in zip(kinds,numbers):
            if os.path.isfile(path+kind+'.csv'):
                  print('files already exist')
            else:
                  with open(path+kind+'.csv','a') as file:
                        file.write('pair,image_1,image_2,x1,y1,z1,x2,y2,z2\n')
                        for i in range(number):
                              random_1, random_2 = np.random.randint(9600),np.random.randint(9600)
                              image_1, image_2 = df.iloc[random_1,0], df.iloc[random_2,0]
                              x1, y1, z1 = df.iloc[random_1,1],df.iloc[random_1,2],df.iloc[random_1,3]
                              x2, y2, z2 = df.iloc[random_2,1],df.iloc[random_2,2],df.iloc[random_2,3]
                              line = 'pair_{},{},{},{},{},{},{},{},{}\n'.format(i,image_1,image_2,x1,y1,z1,x2,y2,z2)
                              file.write(line)
      

def find_mean(dir_path):
      mean = torch.zeros(3)
      df = pd.read_csv(dir_path + 'positions.csv', header=None)
      for i in range(len(df)):
            try:
                  path = dir_path + 'images/' + df.iloc[i,0]
                  image = Image.open(path)
                  image = F.to_tensor(image)[:3,:,:]
                  update = torch.mean(image.view(3,-1),dim=1)
                  mean = i/(i+1.) * mean + 1./(i+1.) * update
            except:
                  print(i)
      return mean

def find_std(dir_path,mean):
      mean_of_diff_squared = torch.zeros(3)
      df = pd.read_csv(dir_path + 'positions.csv', header=None)
      for i in range(len(df)):
            path = dir_path + 'images/' + df.iloc[i,0]
            image = Image.open(path)
            image = F.to_tensor(image)[:3,:,:]
            update = torch.mean(image.view(3,-1),dim=1)
            mean_of_diff_squared = i/(i+1.) * mean_of_diff_squared + 1./(i+1.) * (update-mean)**2
      return mean_of_diff_squared**0.5

def move_forward_facing_images():
      old_path = '/data/cvfs/fml35/own_datasets/grid_world/3d_cube_equidistance'
      old_df = pd.read_csv(old_path + '/images/positions.csv', header=None)
      new_path = '/data/cvfs/fml35/own_datasets/grid_world/3d_cube_one_direction'
      for i in range(2400):
            number = 2+4*i
            old_name = old_df.iloc[number,0]
            x, y, z = old_df.iloc[number,1],old_df.iloc[number,2],old_df.iloc[number,3]
            new_name = 'render_{}_x_{}_y_{}_z_{}.png'.format(i,x,y,z)

            shutil.copy(old_path+'/images/'+old_name,new_path+'/images/'+new_name)
            row = [new_name,x,y,z]
            with open('{}/positions.csv'.format(new_path), 'a') as csv_file:
                  wr = csv.writer(csv_file)
                  wr.writerow(row)



# dir_path = '/data/cvfs/fml35/own_datasets/grid_world/ignas_living_room/'
# mean = find_mean(dir_path)
# print(mean)
# std = find_std(dir_path,mean)
# print(std)

#process_data()
#move_forward_facing_images()