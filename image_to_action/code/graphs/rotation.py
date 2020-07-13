import networkx as nx
import torch
import os

from conversions import Converter

def add_nodes(G, total_number_images,converter):
    for i in range(total_number_images):
        pose = converter.index_to_pose(i)
        G.add_node(i, pose=pose, terminate=False)
        G.add_node(i + total_number_images, pose=pose, terminate=True)


def add_edges(G, move_actions_dict, total_number_images,converter):
    for i in range(total_number_images):
        pose = G.nodes[i]['pose']
        for test_action in move_actions_dict:
            adj_pos = pose + move_actions_dict[test_action]['change']
            adj_pos[3] = adj_pos[3] % 360
            if converter.validate_pose(adj_pos):
                adj_index = converter.pose_to_index(adj_pos) 
                G.add_edge(i, adj_index, action=move_actions_dict[test_action]['name'])
        G.add_edge(i, i+total_number_images, action='term')


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
action_to_index = {'stay':0, 'pos x': 1, 'neg x': 2, 'pos y': 3, 'neg y': 4,
                    'pos z': 5, 'neg z': 6, 'rot +': 7, 'rot -': 8, 'term': 9}

def create_network(move_actions_dict,converter):
    total_number_images = 9600
    G = nx.MultiDiGraph(total_number_images = 9600, rotation=True)
    add_nodes(G,total_number_images)
    add_edges(G,move_actions_dict, total_number_images,converter)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/rotation.gpickle') 


# min_pos = torch.tensor([-1.4,-5.2,1.3,0.])
# max_pos = torch.tensor([0.5,-3.3,1.8,270.])
# steps = torch.tensor([0.1,0.1,0.1,90.])
# converter = Converter(min_pos,max_pos,steps,9600)
# create_network(move_actions_dict,converter)
# G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/rotation.gpickle')
# print(G.nodes[7]['pose'])




    