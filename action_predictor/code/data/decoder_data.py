import torch
from torch.utils import data

import os.path
import networkx as nx
import sys

sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/code"))
sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/code/data"))

class Decoder_Dataset(data.Dataset):
      def __init__(self,graph_path,config):
            'Initialization'
            self.config = config
            self.graph = nx.read_gpickle(graph_path)
            self.number_nodes = self.graph.number_of_nodes()

      def __len__(self):
        return self.config["training"]["pairs_per_epoch"]

      def __getitem__(self, indices=None):

            assert (indices.shape[0] == 2 and indices.dtype == torch.int64), "indices or shape or dtype is wrong"
            pose_1 = self.perturb_pose(self.graph.nodes[indices[0].item()]['pose'])
            pose_2 = self.perturb_pose(self.graph.nodes[indices[1].item()]['pose'])
            poses = torch.cat((pose_1,pose_2)).cuda()
            target_distribution = self.calc_target_distribution(indices)
            return poses, indices, target_distribution

      def perturb_pose(self,pose):
            perturbation = torch.cat((0.1*torch.rand((3,)) - 0.05,torch.zeros((1,)))).cuda()
            perturbed_pose = pose + perturbation
            return perturbed_pose


      def calc_target_distribution(self,indices):
            action_to_index = {'stay':0, 'pos x': 1, 'neg x': 2, 'pos y': 3, 'neg y': 4,
                              'pos z': 5, 'neg z': 6, 'rot +': 7, 'rot -': 8, 'term': 9}

            if indices[0] + int(self.number_nodes/2) == indices[1]:
                  return torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,]).double().cuda()
            else:
                  target_node = indices[1].item()
                  path_lengths = 1000*torch.ones(9).cuda()
                  for adj_node in (self.graph[indices[0].item()]):
                        action_index = action_to_index[self.graph[indices[0].item()][adj_node][0]['action']]
                        if action_index != 9:
                              path_lengths[action_index] = nx.shortest_path_length(self.graph,adj_node,target_node)
                  path_lengths -= torch.min(path_lengths)
                  target_dist = (path_lengths < 0.0001).to(torch.float64)/ torch.sum((path_lengths < 0.0001).to(torch.float64))
            return torch.cat((target_dist,torch.zeros((1,)).double().cuda()))