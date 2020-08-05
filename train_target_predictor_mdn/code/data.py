from torch.utils import data
import json
import torch
import re
import ast


class Target_Predictor_Dataset(data.Dataset):
    def __init__(self, file_path,number_examples):
        self.number_examples = number_examples
        self.target_poses = torch.empty((number_examples,3)).cuda()
        self.normalised_target_poses = torch.empty((number_examples,3)).cuda()
        self.cubes = torch.empty((number_examples,6)).cuda()
        self.descriptions_strings = []
        self.descriptions_lengths = torch.zeros(number_examples).cuda()
        
        with open(file_path, 'r') as csv_file:
            for i in range(number_examples):
                    line = csv_file.readline()
                    result = re.findall('\[(.*?)\]', line)
                    cube = torch.tensor(ast.literal_eval('['+result[0]+']'))
                    target_pose = torch.tensor(ast.literal_eval('['+result[1]+']'))
                    description = line.rsplit(',', 1)[1].rstrip("\n").replace('.','').lower()
                    self.target_poses[i] = target_pose
                    self.cubes[i] = cube
                    self.descriptions_strings.append(description)
                    self.descriptions_lengths[i] = len(description.split())
                    self.normalised_target_poses[i] = target_pose - cube[:3]

        print(self.normalised_target_poses[:10])

    def __len__(self):
        return self.number_examples

    def __getitem__(self,index):
        return self.descriptions_strings[index],self.normalised_target_poses[index]


     