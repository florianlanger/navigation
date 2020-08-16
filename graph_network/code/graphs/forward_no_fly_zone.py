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
        pose = G.nodes[i]['pose']
        if converter.check_flyable_pose(pose):
            G.nodes[i]['flyable'] = True
            G.nodes[i + total_number_non_terminate_nodes]['flyable'] = True
            forward_action = []
            degree_after_mod = (pose[3] / 0.25) % 1.
            if  (-0.001 < degree_after_mod  and degree_after_mod < 0.001) or (0.999 < degree_after_mod  and degree_after_mod < 1.001):
                forward_action.append(orientation_to_forward_action[int((pose[3]*4).round())])
            for test_action in always_valid_actions + forward_action:
                adj_pos = pose + move_actions_dict[test_action]['change']
                adj_pos[3] = adj_pos[3] % 1.
                if 0.999 < adj_pos[3] and adj_pos[3] < 1.001:
                    adj_pos[3] = 0.0
                if converter.validate_pose(adj_pos):
                    if converter.check_flyable_pose(adj_pos):
                        adj_index = converter.pose_to_index(adj_pos) 
                        G.add_edge(i, adj_index, action=move_actions_dict[test_action]['name'])
            G.add_edge(i, i+total_number_non_terminate_nodes, action='term')
        else:
            G.nodes[i]['flyable'] = False
            G.nodes[i+total_number_non_terminate_nodes]['flyable'] = False

def create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter):
    G = nx.MultiDiGraph()
    add_nodes(G,51 * 33 * 25 * 12,converter)
    add_edges(G,move_actions_dict,51 * 33 * 25 * 12,always_valid_actions,orientation_to_forward_action,converter)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/ignas_big_room_fine_grid.gpickle') 


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
    steps = torch.tensor([0.1,0.1,0.1,0.0833333])
    number_poses = 51 * 33 * 25 * 12

    objects =  {
                        "arm chair": {"dimensions":[-0.7,1.1,0.46,0.6,0.6,0.46],"scaling":[1,1,1]},
                        "bookshelf": {"dimensions":[-1.91,0.67,1.74,0.14,0.95,0.35],"scaling":[1,1,1]},
                        "desk chair": {"dimensions":[-1.69,1.96,0.46,0.373,0.24,0.46],"scaling":[1,1,1]},
                        "lamp": {"dimensions":[-0.75,0.67,2.39,0.15,0.13,0.20],"scaling":[1,1,1]},
                        "sideboard": {"dimensions":[-1.78,0.67,0.42,-0.24,0.95,0.40],"scaling":[1,1,1]},
                        "sofa": {"dimensions":[1.1,0.3,0.5,0.55,0.8,0.4],"scaling":[1,1,1]},
                        "couch table": {"dimensions":[-0.87,-0.20,0.20,0.35,0.32,0.21],"scaling":[1,1,1]},
                        "big table": {"dimensions":[2.64,0.68,0.36,0.52,0.49,0.38],"scaling":[1,1,1]},
                        "radiator": {"dimensions":[-0.52,2.23,0.40,0.5,0.07,0.4],"scaling":[1,1,1]},
                        "sink": {"dimensions":[2.16,2.76,0.81,0.2,0.19,0.15],"scaling":[1,1,1]},

                    }



    objects_no_fly = objects_to_no_fly(objects)

    converter = Converter(min_pos,max_pos,steps,number_poses,objects_no_fly)

    create_network(move_actions_dict,always_valid_actions,orientation_to_forward_action,converter)
    G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/ignas_big_room_fine_grid.gpickle')



    # index_start = converter.pose_to_index(torch.tensor([-1.9,-1.,0.,0.]))
    # index_end = converter.pose_to_index(torch.tensor([-1.9,-0.8,0.,0.7]))
    # print(G.nodes[index_start])
    # print(G.nodes[index_end])


    # shortest_path = nx.shortest_path(G,index_start,index_end)

    # for i in range(len(shortest_path)-1):
    #     print(G[shortest_path[i]][shortest_path[i+1]][0]['action'])

    print(G.nodes[19839])

    iterator = G.predecessors(22491)


    print('Should not have predecessor')
    print(next(iterator))
    print('---------')
 

    print(G.edges([19839]))
    index_start = 19839
    index_start_2 = 22491
    index_end = 92341

    print(G[index_start])
    for adj_node in (G[index_start]):
        print(adj_node)

    print(G.nodes[index_start])
    print(G.nodes[index_start_2])
    print(G.nodes[index_end])
    print(G.nodes[19839]["flyable"])


    shortest_path = nx.shortest_path(G,index_start,index_end)

    for i in range(len(shortest_path)-1):
        print(G[shortest_path[i]][shortest_path[i+1]][0]['action'])


