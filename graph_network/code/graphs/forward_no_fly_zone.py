import networkx as nx
import torch
import os
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(os.path.abspath("/Users/legend98/Google Drive/MPhil project/navigation/graph_network/code/graphs"))
from conversions import Converter

def objects_to_no_fly(objects):
    objects_no_fly = torch.zeros((len(objects),2,3))
    for i,key in enumerate(objects):
        objects_no_fly[i,0] = torch.from_numpy(np.array(objects[key]['dimensions'][:3]) - objects[key]['dimensions'][3:6])
        objects_no_fly[i,1] = torch.from_numpy(np.array(objects[key]['dimensions'][:3]) + objects[key]['dimensions'][3:6])
    return objects_no_fly

def add_nodes(G, total_number_non_terminate_nodes,converter):
    for i in range(total_number_non_terminate_nodes):
        pose = converter.index_to_pose(i)
        G.add_node(i, pose=pose, terminate=False)
        G.add_node(i +total_number_non_terminate_nodes, pose=pose, terminate=True)

def add_edges(G, move_actions_dict, total_number_non_terminate_nodes,always_valid_actions,orientation_to_forward_action,converter):
    for i in tqdm(range(total_number_non_terminate_nodes)):
        pose = G.node[i]['pose']
        if converter.check_flyable_pose(pose):
            G.node[i]['flyable'] = True
            G.node[i + total_number_non_terminate_nodes]['flyable'] = True
            forward_action = []
            print(pose)
            print((pose[3] / 0.25) % 1.)
            if (pose[3] / 0.25) % 1. < 0.0001:
                forward_action.append(orientation_to_forward_action[int(pose[3]*4)])
            for test_action in always_valid_actions + forward_action:
                adj_pos = pose + move_actions_dict[test_action]['change']
                adj_pos[3] = adj_pos[3] % 1.
                if converter.validate_pose(adj_pos):
                    if converter.check_flyable_pose(adj_pos):
                        adj_index = converter.pose_to_index(adj_pos) 
                        G.add_edge(i, adj_index, action=move_actions_dict[test_action]['name'])
            G.add_edge(i, i+total_number_non_terminate_nodes, action='term')
        else:
            G.node[i]['flyable'] = False
            G.node[i+total_number_non_terminate_nodes]['flyable'] = False

def create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter):
    G = nx.MultiDiGraph()
    add_nodes(G, 8 * 4,converter)
    add_edges(G,move_actions_dict,8 * 4,always_valid_actions,orientation_to_forward_action,converter)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/test.gpickle') 


if __name__ == "__main__":
      
    #first list are those that are valid when orientation is 0, i.e. stay, pos y, pos z ,neg z , rot +, rot - (term is handled separately)
    always_valid_actions = [0,5,6,7,8]

    move_actions_dict = {0:{'name':'stay','change':torch.tensor([0.,0.,0.,0.])},
                    1:{'name':'pos x','change':torch.tensor([0.1,0.,0.,0.])},
                    2:{'name':'neg x','change':torch.tensor([-0.1,0,0,0.])},
                    3:{'name':'pos y','change':torch.tensor([0,0.1,0,0.])},
                    4:{'name':'neg y','change':torch.tensor([0.,-0.1,0,0.])},
                    5:{'name':'pos z','change':torch.tensor([0,0,0.1,0.])},
                    6:{'name':'neg z','change':torch.tensor([0,0,-0.1,0.])},
                    7:{'name':'rot +','change':torch.tensor([0,0,0.,0.0833333])},
                    8:{'name':'rot -','change':torch.tensor([0,0,0.,-0.0833333])}
                        }

    orientation_to_forward_action = {0:1,1:3,2:2,3:4}


    min_pos = torch.tensor([-1.9,-1.,0.,0.])
    max_pos = torch.tensor([3.1,2.2,2.4,0.91666666])
    steps = torch.tensor([0.2,0.2,0.2,0.0833333])
    number_poses = 504900

    objects = {
                    "drawer": {"dimensions": [4.1,0.29,0.59,0.6,0.29,0.59],"scaling": [1,1,1]},
                    "couch": {"dimensions": [2.13,2.22,0.5,0.45,1.02,0.5],"scaling": [1,1,1]},
                    "bed": {"dimensions": [3.58,2.33,0.31,1.0,0.93,0.31],"scaling": [1,1,1]},
                    "desk": {"dimensions": [3.63,4.43,0.39,0.55,0.25,0.39],"scaling": [1,1,1]},
                    "lamp": {"dimensions": [4.54,3.64,0.70,0.1,0.1,0.24],"scaling": [1,1,1]},
                    "bedside table": {"dimensions": [4.52,3.6,0.23,0.2,0.2,0.23],"scaling": [1,1,1]},
                    "computer screen": {"dimensions": [3.62,4.62,1.09,0.32,0.12,0.31],"scaling": [1,1,1]},
                    "box": {"dimensions": [0.27,2.38,0.19,0.27,0.48,0.19],"scaling": [1,1,1]}
                }



    objects_no_fly = objects_to_no_fly(objects)

    converter = Converter(min_pos,max_pos,steps,number_poses,objects_no_fly)

    create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter)
    G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/test.gpickle')


    print(converter.index_to_pose(0))
    print(converter.index_to_pose(23 * 8 * 4 +1))
    print(converter.index_to_pose(2*23 * 4 * 8))
    index_start = converter.pose_to_index(torch.tensor([1.2671, 2.6516, 0.6030, 0.7299]))
    #
    index_end = converter.pose_to_index(torch.tensor([3.8, 3.8, 1.8, 0.0]))
    print(G.node[index_start])


    shortest_path = nx.shortest_path(G,index_start,index_end)

    for i in range(len(shortest_path)-1):
        print(G[shortest_path[i]][shortest_path[i+1]][0]['action'])


