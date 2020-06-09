import torch
import torchvision
from torch.utils import data
import pandas as pd 
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
import csv



class Pose_Dataset(data.Dataset):
      def __init__(self, dir_path, file_name, config, transform=None, kind="training"):
            'Initialization'
            self.dir_path = dir_path
            self.df = pd.read_csv(dir_path+'/'+file_name,quoting=csv.QUOTE_NONNUMERIC,header=None)
            self.config = config
            self.transform = transform
            self.kind = kind

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

      def __getitem__(self, indices):
            if indices.shape[0] != 1 or indices.dtype != torch.int64:
                print(indices.dtype)
                raise Exception("One index of type 'torch.int64' has to be specified")
            image_name = self.df.iloc[indices[0].item(),0]
            # positions: x1,y1,z1,x2,y2,z2
            position = torch.from_numpy(self.df.iloc[indices[0].item(),1:].to_numpy().astype(np.float64)).cuda()
            #divide angle by 90
            position[3] = position[3] / 360.
            image = Image.open(self.dir_path +'/images/' + image_name)
            if self.transform:
                  image = self.transform(image)
            image = F.to_tensor(image)[:3,:,:].cuda()
            
            occluded = False
            if torch.rand(1).item() <  self.config["data"]["occlusion_probability"]:
                image = self.occlud(image)
                occluded = True
            

            return image, position, occluded

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

