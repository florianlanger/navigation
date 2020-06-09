import networkx as nx
import torch
import os



def calc_action_penalties_hard_coded(positions, terminate_penalty):
      action_penalties = torch.zeros(8).cuda()
      steps = torch.tensor([[0.,0.,0.],[0.1,0.,0.],[-0.1,0.,0.],[0.,0.1,0.],[0.,-0.1,0.],[0.,0.,.1],[0.,0.,-0.1]])
      for i,step in enumerate(steps):
            action_penalties[i] = torch.sum(torch.abs(positions[:3] + step - positions[3:6]))
      action_penalties = action_penalties - torch.min(action_penalties[:7])
      action_penalties[7] = terminate_penalty

      if torch.norm(positions[:3]-positions[3:6]) < 0.00001:
            action_penalties[7] = 0
            # Ensure that when at goal move away have loss 0.2, and when stay 
            action_penalties *= 2
            action_penalties[0] = 0.1
      return action_penalties


def calc_action_penalties(positions,indices,graph,config):
    if config["model"]["number_outputs"] == 8:
        #return calc_action_penalties_hard_coded(positions, config["loss"]["terminate_penalty"])
        return calc_action_penalties_graph(indices,graph,config)
    elif config["model"]["number_outputs"] == 10:
        return calc_action_penalties_graph(indices,graph,config)


def calc_action_penalties_graph(indices,graph,config):
    if config["model"]["number_outputs"] == 8:
        action_to_index = {'stay':0, 'pos x': 1, 'neg x': 2, 'pos y': 3, 'neg y': 4,'pos z': 5, 'neg z': 6, 'term': 7}
    elif config["model"]["number_outputs"] == 10:
        action_to_index = {'stay':0, 'pos x': 1, 'neg x': 2, 'pos y': 3, 'neg y': 4,
                    'pos z': 5, 'neg z': 6, 'rot +': 7, 'rot -': 8, 'term': 9}

    action_penalties =  -1000*torch.ones(config["model"]["number_outputs"]).cuda()
    if indices[0] == indices[1]:
        action_penalties[-1] = 0.
    for adj_node in (graph[indices[0].item()]):
        # Try out all nodes except terminate one
        if adj_node < config["data"]["number_images"]:
            action_index = action_to_index[graph[indices[0].item()][adj_node][0]['action']]
            target_node = indices[1].item() + config["data"]["number_images"]
            action_penalties[action_index] = nx.shortest_path_length(graph,adj_node,target_node)
    action_penalties[action_penalties==-1000] = torch.max(action_penalties)
    action_penalties = (action_penalties - torch.min(action_penalties)) * 0.1

    if indices[0] != indices[1]:
        action_penalties[-1] = config["loss"]["terminate_penalty"]

    return action_penalties

def calc_action_penalties_occluded(config):
    action_penalties = 0.2 * torch.ones(config["model"]["number_outputs"]).cuda()
    action_penalties[7] = 0
    action_penalties[8] = 0
    action_penalties[9] = config["loss"]["terminate_penalty"]
    return action_penalties


    