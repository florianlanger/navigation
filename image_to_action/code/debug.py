import torch
import os
import networkx as nx

from data.action_penalties import calc_action_penalties_graph
from utilities import load_config
from graphs.rotation import pose_to_index

config = load_config('/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/3d_grid_equidistance/config.json')

graph_path = os.path.dirname(os.path.realpath(__file__)) + '/../graphs/forward_department.gpickle'
G = nx.read_gpickle(graph_path)

index_start = pose_to_index(torch.tensor([0.1, -4.5, 1.7, 270.0])) 
index_goal = pose_to_index(torch.tensor([0.2, -3.6, 1.7, 0.0])) 
print(index_start)
print(index_goal)

print(G[index_start])
# print(nx.shortest_path_length(G,4126,2350 + 9600))

print(calc_action_penalties_graph(torch.tensor([index_start, index_goal]),G, config))