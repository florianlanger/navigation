import torch
import torchvision
from torch.utils import data
import pandas as pd 
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import csv



class Pose_Dataset(data.Dataset):
      def __init__(self, dir_path, file_name, transform=None, kind="training"):
            'Initialization'
            self.dir_path = dir_path
            self.df = pd.read_csv(dir_path+'/'+file_name,quoting=csv.QUOTE_NONNUMERIC,header=None)
            self.length = len(self.df)
            self.transform = transform
            self.kind = kind

      def __len__(self):
        return self.length

      def __getitem__(self, index):
            image_name = self.df.iloc[index,0]
            # positions: x1,y1,z1,x2,y2,z2
            pose = torch.from_numpy(self.df.iloc[index,1:].to_numpy().astype(np.float64)).cuda()
            pose[3] = pose[3] / 360.
            image = Image.open(self.dir_path +'/images/' + image_name)
            if self.transform:
                  image = self.transform(image)
            image = F.to_tensor(image)[:3,:,:].cuda()
            return image, pose

