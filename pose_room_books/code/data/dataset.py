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
      def __init__(self, path_file, path_image_folder, config, transform=None, kind="training"):
            'Initialization'
            self.path_image_folder = path_image_folder
            with open(path_file, 'r') as fp:
                  self.data_dict = json.load(fp)
            self.config = config
            self.transform = transform
            self.kind = kind
            self.number_images = len(self.data_dict)
            self.poses = torch.zeros((self.number_images,7)).cuda()
            for i in range(self.number_images):
                  image_name = 'facing_bookshelf_part2_frame_' + str((i + 1)*10).zfill(5) + '.png'
                  # divide positions by 300 !!!
                  self.poses[i,:3] = torch.from_numpy(np.array(self.data_dict[image_name]['translation'])) / 300.
                  self.poses[i,3:7] = torch.from_numpy((R.from_rotvec(self.data_dict[image_name]['rotation']).as_quat()))


      def __len__(self):
        return self.number_images

      def __getitem__(self, index):

            image_name = 'facing_bookshelf_part2_frame_' + str((index + 1)*10).zfill(5) + '.png' 
            image = Image.open(self.path_image_folder + '/' + image_name)
            if self.transform:
                  image = self.transform(image)
            image = F.to_tensor(image)[:3,:,:].cuda()
            

            return image, self.poses[index]

