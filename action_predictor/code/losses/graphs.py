import networkx as nx
import numpy as np
import torch
import os

from graphs.no_rotation import index_to_position_no_orientation

# config = {"model":{"number_outputs":8}}
# indices = torch.tensor([[382,24],[957,162],[222,850]])
# positions = torch.empty(3,6)
# for i,pair in enumerate(indices):
#     for j,index in enumerate(pair):
#         positions[i,3*j:3*(j+1)] = torch.tensor(index_to_position_no_orientation(index))
# terminate_penalty = 10

def calc_action_penalties_graph(indices,config, terminate_penalty):
    if config["model"]["number_outputs"] == 8:
        action_to_index = {'stay':0, 'pos x': 1, 'neg x': 2, 'pos y': 3, 'neg y': 4,'pos z': 5, 'neg z': 6, 'term': 7}
        graph_path = os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/no_rotation.gpickle'
    elif config["model"]["number_outputs"] == 10:
        graph_path = os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/rotation.gpickle'
        action_to_index = {'stay':0, 'pos x': 1, 'neg x': 2, 'pos y': 3, 'neg y': 4,
                    'pos z': 5, 'neg z': 6, 'rot +': 7, 'rot -': 8, 'term': 9}
    G = nx.read_gpickle(graph_path)
    total_number_images = config["data"]["number_images"]
    n_actions = config["model"]["number_outputs"]
    action_penalties = torch.empty(indices.shape[0],n_actions).cuda()
    for i,pair in enumerate(indices):
        penalties =  -1000*torch.ones(n_actions).cuda()
        if pair[0] == pair[1]:
            penalties[-1] = 0.
        for adj_node in (G[pair[0].item()]):
            # Try out all nodes except terminate one
            if adj_node < total_number_images:
                action_index = action_to_index[G[pair[0].item()][adj_node][0]['action']]
                penalties[action_index] = nx.shortest_path_length(G,adj_node,pair[1].item() + total_number_images)
        penalties[penalties==-1000] = torch.max(penalties)
        penalties = (penalties - torch.min(penalties)) * 0.1

        if pair[0] != pair[1]:
            penalties[-1] = terminate_penalty

        action_penalties[i] = penalties

    return action_penalties

    