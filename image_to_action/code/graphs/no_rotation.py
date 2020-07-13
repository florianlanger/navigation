import networkx as nx
import torch
import os

from conversions import index_to_position, position_to_index, validate_position


def add_nodes_no_rotation(G, total_number_images):
    for i in range(total_number_images):
        G.add_node(i, position=index_to_position(i), terminate=False)
        G.add_node(i + total_number_images, position=index_to_position(i), terminate=True)

def add_edges_no_rotation(G, move_actions_dict, total_number_images):
    for i in range(total_number_images):
        position = G.nodes[i]['position']
        for test_action in move_actions_dict:
            adj_pos = position + move_actions_dict[test_action]['change']
            if validate_position(adj_pos):
                adj_index = position_to_index(adj_pos) 
                G.add_edge(i, adj_index, action=move_actions_dict[test_action]['name'])
        G.add_edge(i, i+total_number_images, action='term')


move_actions_dict = {0:{'name':'stay','change':torch.tensor([0.,0.,0.])},
                1:{'name':'pos x','change':torch.tensor([0.1,0.,0.])},
                2:{'name':'neg x','change':torch.tensor([-0.1,0,0])},
                3:{'name':'pos y','change':torch.tensor([0,0.1,0])},
                4:{'name':'neg y','change':torch.tensor([0.,-0.1,0])},
                5:{'name':'pos z','change':torch.tensor([0,0,0.1])},
                6:{'name':'neg z','change':torch.tensor([0,0,-0.1])}
                    }


def create_network_no_rotation(move_actions_dict):
    total_number_images = 2400
    G = nx.MultiDiGraph(total_number_images = 2400, rotation=False)
    add_nodes_no_rotation(G,total_number_images)
    add_edges_no_rotation(G,move_actions_dict, total_number_images)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/no_rotation.gpickle') 

create_network_no_rotation(move_actions_dict)
G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/no_rotation.gpickle')
print(G[3])



    