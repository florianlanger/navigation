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
            if (pose[3] / 0.25) % 1. < 0.00001:
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
    add_nodes(G,13984,converter)
    add_edges(G,move_actions_dict,13984,always_valid_actions,orientation_to_forward_action,converter)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/room_books_forward_hollow_table.gpickle') 


if __name__ == "__main__":
      
    #first list are those that are valid when orientation is 0, i.e. stay, pos y, pos z ,neg z , rot +, rot - (term is handled separately)
    always_valid_actions = [0,5,6,7,8]

    move_actions_dict = {0:{'name':'stay','change':torch.tensor([0.,0.,0.,0.])},
                    1:{'name':'pos x','change':torch.tensor([0.2,0.,0.,0.])},
                    2:{'name':'neg x','change':torch.tensor([-0.2,0,0,0.])},
                    3:{'name':'pos y','change':torch.tensor([0,0.2,0,0.])},
                    4:{'name':'neg y','change':torch.tensor([0.,-0.2,0,0.])},
                    5:{'name':'pos z','change':torch.tensor([0,0,0.2,0.])},
                    6:{'name':'neg z','change':torch.tensor([0,0,-0.2,0.])},
                    7:{'name':'rot +','change':torch.tensor([0,0,0.,0.25])},
                    8:{'name':'rot -','change':torch.tensor([0,0,0.,-0.25])}
                        }

    orientation_to_forward_action = {0:1,1:3,2:2,3:4}


    min_pos = torch.tensor([0.2,0.2,0.2,0.])
    max_pos = torch.tensor([3.8,4.6,1.6,0.75])
    steps = torch.tensor([0.2,0.2,0.2,0.25])
    number_poses = 13984

    objects =  {
                        "printer": {"dimensions":[3.75,0.17,0.15,0.2,0.15,0.15],"scaling":[1,1,1]},
                        "bench": {"dimensions":[1.06,2.18,0.23,0.24,0.72,0.23],"scaling":[0.8,0.5,1]},
                        "big table": {"dimensions":[1.66,2.11,0.4,0.36,0.79,0.4],"scaling":[0.7,0.65,2.4]},
                        "side table": {"dimensions":[1.86,3.25,0.75,0.48,0.32,0.05],"scaling":[0.7,0.7,2.4]},
                        "arm chair": {"dimensions":[3.30,4.40,0.5,0.45,0.45,0.5],"scaling":[0.7,0.8,1]},
                        "chess board table": {"dimensions":[3.56,2.81,0.19,0.19,0.19,0.19],"scaling":[0.8,0.8,1]},
                        "shelf with the nespresso box": {"dimensions":[3.87,2.82,1.03,0.13,0.36,0.17],"scaling":[1,0.8,1]},
                        "chair": {"dimensions":[2.10,2.35,0.51,0.3,0.35,0.51],"scaling":[1,1,1]},
                        "couch": {"dimensions":[1.85,-0.9,0.39,1.37,0.75,0.39],"scaling":[0.5,0.6,1]},
                        "computer screen": {"dimensions":[1.45,1.59,1.08,0.15,0.25,0.28],"scaling":[0.9,0.8,0.8]},
                        "box with newspapers": {"dimensions":[1.20,3.25,0.5,0.22,0.22,0.5],"scaling":[0.8,0.8,0.8]}
                    }



    objects_no_fly = objects_to_no_fly(objects)

    converter = Converter(min_pos,max_pos,steps,number_poses,objects_no_fly)

    create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter)
    G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/room_books_forward_hollow_table.gpickle')



    index_start = converter.pose_to_index(torch.tensor([0.4,2.,0.6,0.]))
    index_end = converter.pose_to_index(torch.tensor([3.,2.,0.6,0.]))
    print(G.node[index_start])
    print(G.node[index_end])


    shortest_path = nx.shortest_path(G,index_start,index_end)

    for i in range(len(shortest_path)-1):
        print(G[shortest_path[i]][shortest_path[i+1]][0]['action'])


