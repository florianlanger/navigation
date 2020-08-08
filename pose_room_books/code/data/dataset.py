import torch
import torchvision
from torch.utils import data
import pandas as pd 
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import csv
from scipy.spatial.transform import Rotation as R
import json


class Pose_Dataset(data.Dataset):
      def __init__(self, path_file, path_image_folder, config, transform=None, kind="training",max_number_images=None):
            'Initialization'
            self.path_image_folder = path_image_folder
            with open(path_file, 'r') as fp:
                  self.data_dict = json.load(fp)
            self.config = config
            self.transform = transform
            self.kind = kind
            if max_number_images is not None:
                  self.number_images = max_number_images
            else:
                  self.number_images = len(self.data_dict)
            self.poses = torch.zeros((self.number_images,4)).cuda()
            self.image_names = []
            for i,key in enumerate(self.data_dict):
                  if i == max_number_images:
                        break 
                  self.image_names.append(key)
                  # divide positions by 1000 !!!
                  self.poses[i,:3] = torch.from_numpy(np.array(self.data_dict[key]['translation'])) / 1000.
                  self.poses[i,3] = float(self.data_dict[key]['z_after_crop'])/360


      def __len__(self):
        return self.number_images

      def __getitem__(self, index):

            image_name = self.image_names[index]
            image = Image.open(self.path_image_folder + '/' + image_name)
            if self.transform:
                  image = self.transform(image)
            image = F.to_tensor(image)[:3,:,:].cuda()
            
            

            return image, self.poses[index],image_name

