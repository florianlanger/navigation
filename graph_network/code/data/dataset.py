import torch
from torch.utils import data
import os.path
import networkx as nx
import sys

class Graph_Replacer_Dataset(data.Dataset):
      def __init__(self,graph_path,config):
            'Initialization'
            self.config = config
            self.graph = nx.read_gpickle(graph_path)
            self.number_nodes = self.graph.number_of_nodes()

      def __len__(self):
        return self.config["training"]["pairs_per_epoch"]

      def __getitem__(self, indices=None):

            assert (indices.shape[0] == 2 and indices.dtype == torch.int64), "indices or shape or dtype is wrong"
            pose_1 = self.graph.nodes[indices[0].item()]['pose']
            pose_2 = self.graph.nodes[indices[1].item()]['pose']
            poses = torch.cat((pose_1,pose_2)).cuda()
            number_steps = torch.tensor([nx.shortest_path_length(self.graph,indices[0].item(),indices[1].item())]).cuda()
            return poses,indices, number_steps
