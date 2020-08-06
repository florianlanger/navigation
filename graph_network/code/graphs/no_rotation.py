import networkx as nx
import torch
import os
from tqdm import tqdm

# from conversions import index_to_position, position_to_index, validate_position


def add_nodes_no_rotation(G, total_number_images,scene,objects):
    for i in range(total_number_images):
        G.add_node(i, position=index_to_position(i,scene))


def add_edges_no_rotation(G, move_actions_dict, total_number_images,scene,objects):
    for i in tqdm(range(total_number_images)):
        position = G.node[i]['position']
        for test_action in move_actions_dict:
            adj_pos = position + move_actions_dict[test_action]['change']
            if validate_position(adj_pos,scene,objects):
                adj_index = position_to_index(adj_pos,scene) 
                G.add_edge(i, adj_index, action=move_actions_dict[test_action]['name'])


move_actions_dict = {0:{'name':'stay','change':torch.tensor([0.,0.,0.])},
                1:{'name':'pos x','change':torch.tensor([0.2,0.,0.])},
                2:{'name':'neg x','change':torch.tensor([-0.2,0,0])},
                3:{'name':'pos y','change':torch.tensor([0,0.2,0])},
                4:{'name':'neg y','change':torch.tensor([0.,-0.2,0])},
                5:{'name':'pos z','change':torch.tensor([0,0,0.2])},
                6:{'name':'neg z','change':torch.tensor([0,0,-0.2])}
                    }

objects = {
    'big cupboard': torch.tensor([0.63,1.23,2.65,0,2.42,0]),
    'sideboard': torch.tensor([1.2,0.6,0.75,0,0,0]),
    'table': torch.tensor([1.8,0.9,0.2,1.2,0,0.65]),
    'couch': torch.tensor([1.56,0.88,0.7,2.38,2.78,0]),
    'stool': torch.tensor([0.6,0.4,0.4,2.38,2.18,0]),
    'small cupboard': torch.tensor([0.34,0.44,0.7,3.74,2.2,0.]),
    'printer': torch.tensor([0.5,0.5,0.4,3.4,1.25,0]),
    'lamp': torch.tensor([0.25,0.85,0.7,2.28,0,0.75])
}

scene = torch.tensor([3.8,3.2,1.6,0.2,0.2,0.4])

def validate_position(position,scene,objects):
    if torch.all(position + 0.0001 >= scene[3:6]) and torch.all(position[:3] - 0.0001 <= scene[:3] + scene[3:6]):
        for i,key in enumerate(objects):
            dims = objects[key]
            if torch.all(position + 0.0001 >= dims[3:6]) and torch.all(position[:3] - 0.0001 <= dims[:3] + dims[3:6]):
                return False
        
        return True
                
    else:
        return False


def index_to_position(index,scene):
    if index < 20 * 17 * 9 and index >= 0:
        index_each_direction = index * torch.tensor([1.,1.,1.])
        index_each_direction = (index_each_direction % torch.tensor([20.*9*17,9*17,9])// torch.tensor([17*9,9,1]))
        position = scene[3:6] + torch.tensor([0.2,0.2,0.2]) * index_each_direction
        return position
    else:
        raise Exception('Not a valid index')

def position_to_index(position,scene):
        print(position)
        indices = torch.round((position - scene[3:6])/torch.tensor([0.2,0.2,0.2]))
        index = torch.dot(indices,torch.tensor([17*9,9.,1]))
        print(indices)
        print(index)
        if index <  20 * 17 * 9 and index >= 0:
            return int(index.item())
        else:
            raise Exception('Not a valid pose')


def create_network_no_rotation(move_actions_dict,scene,objects):
    total_number_nodes = 20 * 17 * 9
    G = nx.MultiDiGraph()
    add_nodes_no_rotation(G,total_number_nodes,scene,objects)
    add_edges_no_rotation(G,move_actions_dict, total_number_nodes,scene,objects)
    nx.write_gpickle(G,os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/own_room_no_rotation.gpickle') 

# create_network_no_rotation(move_actions_dict,scene,objects)
# G = nx.read_gpickle(os.path.dirname(os.path.realpath(__file__)) + '/../../graphs/own_room_no_rotation.gpickle')
# shortest_path = nx.shortest_path(G,10,40)
# print(G.node[10]['position'])
# print(G.node[40]['position'])
# for i in range(len(shortest_path)-1):
#     print(G[shortest_path[i]][shortest_path[i+1]][0]['action'])

# print(validate_position(torch.tensor([0.4,2.,1.0]),scene,objects))
# print(validate_position(torch.tensor([0.4,2.,2.4]),scene,objects))
# print(validate_position(torch.tensor([0.4,.2,0.3]),scene,objects))
# print(validate_position(torch.tensor([1.5,1.3,1.0]),scene,objects))

# index = position_to_index(torch.tensor([1.6,0.4,0.8]),scene)
# print(index_to_position(index,scene))



    