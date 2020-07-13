import networkx as nx
import torch
import os

from conversions import Converter
from rotation import add_nodes


def add_edges(G, move_actions_dict, total_number_images,always_valid_actions,orientation_to_forward_action,converter):
    for i in range(total_number_images):
        pose = G.nodes[i]['pose']
        for test_action in always_valid_actions + [orientation_to_forward_action[int(pose[3]/90)]]:
            adj_pos = pose + move_actions_dict[test_action]['change']
            adj_pos[3] = adj_pos[3] % 360
            if converter.validate_pose(adj_pos):
                adj_index = converter.pose_to_index(adj_pos) 
                G.add_edge(i, adj_index, action=move_actions_dict[test_action]['name'])
        G.add_edge(i, i+total_number_images, action='term')

    
#first list are those that are valid when orientation is 0, i.e. stay, pos y, pos z ,neg z , rot +, rot - (term is handled separately)
always_valid_actions = [0,5,6,7,8]

move_actions_dict = {0:{'name':'stay','change':torch.tensor([0.,0.,0.,0.])},
                1:{'name':'pos x','change':torch.tensor([0.1,0.,0.,0.])},
                2:{'name':'neg x','change':torch.tensor([-0.1,0,0,0.])},
                3:{'name':'pos y','change':torch.tensor([0,0.1,0,0.])},
                4:{'name':'neg y','change':torch.tensor([0.,-0.1,0,0.])},
                5:{'name':'pos z','change':torch.tensor([0,0,0.1,0.])},
                6:{'name':'neg z','change':torch.tensor([0,0,-0.1,0.])},
                7:{'name':'rot +','change':torch.tensor([0,0,0.,90.])},
                8:{'name':'rot -','change':torch.tensor([0,0,0.,-90.])}
                    }

orientation_to_forward_action = {0:3,1:2,2:4,3:1}

def create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter):
    G = nx.MultiDiGraph(total_number_images = converter.number_poses)
    add_nodes(G,converter.number_poses,converter)
    add_edges(G,move_actions_dict, converter.number_poses,always_valid_actions,orientation_to_forward_action,converter)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/forward_department.gpickle') 


# min_pos = torch.tensor([-1.4,-5.2,1.3,0.])
# max_pos = torch.tensor([0.5,-3.3,1.8,270.])
# steps = torch.tensor([0.1,0.1,0.1,90.])
# number_poses = 9600
# converter = Converter(min_pos,max_pos,steps,number_poses)
# create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter)
# G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/forward_department.gpickle')
# print(G[406])
# print(G.nodes[406]['pose'])
# print(G[4236])
# print(G.nodes[4236]['pose'])
# print(G.edges([406, 410]))
# print(nx.shortest_path_length(G,406,410))
# print(nx.shortest_path_length(G,406,4236))

    