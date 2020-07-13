import networkx as nx
import torch
import os
from tqdm import tqdm

from conversions import Converter


def add_nodes(G, total_number_non_terminate_nodes,converter):
    for i in range(total_number_non_terminate_nodes):
        pose = converter.index_to_pose(i)
        G.add_node(i, pose=pose, terminate=False)
        G.add_node(i +total_number_non_terminate_nodes, pose=pose, terminate=True)

def add_edges(G, move_actions_dict, total_number_non_terminate_nodes,always_valid_actions,orientation_to_forward_action,converter):
    for i in tqdm(range(total_number_non_terminate_nodes)):
        pose = G.nodes[i]['pose']
        if converter.check_flyable_pose(pose):
            G.nodes[i]['flyable'] = True
            G.nodes[i + total_number_non_terminate_nodes]['flyable'] = True
            forward_action = []
            if (pose[3] / 0.25) % 1. < 0.00001:
                forward_action.append(orientation_to_forward_action[int(pose[3]*4)])
            for test_action in always_valid_actions + forward_action:
                adj_pos = pose + move_actions_dict[test_action]['change'].cuda()
                adj_pos[3] = adj_pos[3] % 1.
                if converter.validate_pose(adj_pos):
                    if converter.check_flyable_pose(adj_pos):
                        adj_index = converter.pose_to_index(adj_pos) 
                        G.add_edge(i, adj_index, action=move_actions_dict[test_action]['name'])
            G.add_edge(i, i+total_number_non_terminate_nodes, action='term')
        else:
            G.nodes[i]['flyable'] = False
            G.nodes[i+total_number_non_terminate_nodes]['flyable'] = False

    
#first list are those that are valid when orientation is 0, i.e. stay, pos y, pos z ,neg z , rot +, rot - (term is handled separately)
always_valid_actions = [0,5,6,7,8]

move_actions_dict = {0:{'name':'stay','change':torch.tensor([0.,0.,0.,0.])},
                1:{'name':'pos x','change':torch.tensor([0.1,0.,0.,0.])},
                2:{'name':'neg x','change':torch.tensor([-0.1,0,0,0.])},
                3:{'name':'pos y','change':torch.tensor([0,0.1,0,0.])},
                4:{'name':'neg y','change':torch.tensor([0.,-0.1,0,0.])},
                5:{'name':'pos z','change':torch.tensor([0,0,0.1,0.])},
                6:{'name':'neg z','change':torch.tensor([0,0,-0.1,0.])},
                7:{'name':'rot +','change':torch.tensor([0,0,0.,0.0625])},
                8:{'name':'rot -','change':torch.tensor([0,0,0.,-0.0625])}
                    }

orientation_to_forward_action = {0:3,1:2,2:4,3:1}

def create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter):
    G = nx.MultiDiGraph()
    add_nodes(G,30,converter)
    add_edges(G,move_actions_dict,30,always_valid_actions,orientation_to_forward_action,converter)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/debug_no_fly_living_room_small_angles.gpickle') 


# min_pos = torch.tensor([-1.3,-0.5,0.2,0.]).cuda()
# max_pos = torch.tensor([1.8,1.4,1.7,0.9375]).cuda()
# steps = torch.tensor([0.1,0.1,0.1,0.0625]).cuda()
# number_poses = 163840
# corners_no_fly_zone = torch.tensor([[[0.5,-0.5,0.1],[1.7,1.1,0.9]],[[-1.3,0.5,0.1],[-0.1,1.7,1.1]]]).cuda()
# converter = Converter(min_pos,max_pos,steps,number_poses,corners_no_fly_zone)

# create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter)
G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/debug_no_fly_living_room_small_angles.gpickle')


print(G.nodes[0])
print(G.nodes[27])
print(nx.shortest_path_length(G,27,0))


