import torch
import torchvision
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models import vgg11
from datetime import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from absl import flags
from absl import app
import cv2
import sys
import networkx as nx

sys.path.append(os.path.abspath("/home/mlmi-2019/fml35/Documents/mphil_project/experiments_all/navigation/pose/code"))
from data.dataset import Pose_Dataset
from data.sampler import Sampler
from graphs.conversions import Converter
from utilities import write_to_file, load_config, load_model,load_data_set
from visualisations.plots import visualise_pairs

FLAGS = flags.FLAGS
flags.DEFINE_string("name", None, "name of the experiment")
flags.DEFINE_string("weights", 'best_train_model.pth', "which model weights to load")
flags.DEFINE_integer("number_tests", 5, "The number of tests to do")

def check_not_done(pos_index,move,history_dict):
    if pos_index in history_dict['past_actions']:
        if move in history_dict['past_actions'][pos_index]:
            return False
        else:
            return True
    else:
        return True

def update_history_dict(pos_index,current_pos,output,move,history_dict):
    if pos_index in history_dict['past_actions']:
        history_dict['past_actions'][pos_index].append(move)
    else:
        history_dict['past_actions'][pos_index] = [move]
    
    history_dict['trajectory'].append(list(current_pos.numpy()))
    history_dict['moves'].append(move)
    history_dict['predictions'].append(list(np.round(output.cpu().numpy(),6)))


def find_next_move(images, pos_index, current_pos, network, history_dict, converter):
    output = network(images.unsqueeze(0))[0]
    # moves recommended by networking ordered by descending probability
    _,move_numbers = torch.sort(output,descending=True)
    # Go through moves in order how they were recommended.
    # If move is allowed and has not been done in this pos before, break and do the move
    move = None
    for test_move in move_numbers:
        test_move = test_move.item()
        test_pose = current_pos.clone() + converter.move_to_coords[test_move]
        if converter.validate_pose(converter.map_pose(test_pose)) and check_not_done(pos_index,test_move,history_dict):
            move = test_move
            break
    return output,move

def initialise_history_dict():
    history_dict = {}
    history_dict['past_actions'] = {}
    history_dict['trajectory'] = []
    history_dict['predictions'] = []
    history_dict['moves'] = []
    history_dict['terminated'] = False
    history_dict['statistics'] = {}
    return history_dict

def plot_trajectory(path_to_history_dict,folder_path,config,converter):
    history_dict = np.load(path_to_history_dict,allow_pickle='TRUE').item()
    fig = plt.figure()

    ax = Axes3D(fig)
    x,y,z = history_dict['trajectory'][:,0],history_dict['trajectory'][:,1],history_dict['trajectory'][:,2]
    x_t,y_t,z_t = history_dict['target'][0],history_dict['target'][1],history_dict['target'][2]
    x_s,y_s,z_s = history_dict['trajectory'][0,0],history_dict['trajectory'][0,1],history_dict['trajectory'][0,2]
    #ax.plot(x,y,z)
    ax.scatter(x,y,z,linewidth=1)
    ax.scatter(x_t,y_t,z_t,label='Target')
    ax.plot([x_t,x_t],[y_t,y_t],[converter.min_position[2].item(),z_t],'--')
    ax.scatter(x_s,y_s,z_s,label='Start')
    ax.plot([x_s,x_s],[y_s,y_s],[converter.min_position[2].item(),z_s],'--')
    orientation_to_direction = {0:[0,1],1:[-1,0],2:[0,-1],3:[1,0]}
    for i in range(len(x)):
        ax.text(x[i],y[i],z[i],str(i+1))
    for i in range(len(x)):
        dx, dy = orientation_to_direction[int(history_dict['trajectory'][i,3]//90)]
        ax.quiver(x[i],y[i],z[i], dx, dy, 0, length=0.1, color="green",arrow_length_ratio=0.6)
    # target
    dx_t, dy_t = orientation_to_direction[int(history_dict['target'][3]//90)]
    ax.quiver(x_t,y_t,z_t, dx_t, dy_t, 0, length=0.1, color="orange",arrow_length_ratio=0.6)
    ax.legend()
    if config["data"]["place"] == "living_room":
        ax.set_xlabel('x - windows')
        ax.set_ylabel('y - kitchen')
        ax.set_zlabel('z')
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    ax.set_xlim(converter.min_position[0].item(),converter.max_position[0].item())
    ax.set_ylim(converter.min_position[1].item(),converter.max_position[1].item())
    ax.set_zlim(converter.min_position[2].item(),converter.max_position[2].item())
    if config["graph"]["no_fly"]:
        ax = plot_no_fly(ax,converter.corners_no_fly_zone)
    for angles,name in zip([[0.,270],[-90.,270],[90.,270]],['normal','side','top_down']):
        fig.savefig('{}/plot_{}.png'.format(folder_path,name),dpi=150)
        ax.view_init(elev=angles[0], azim=angles[1])
    plt.close(fig)
    right_images = cv2.vconcat([cv2.imread(folder_path + '/plot_side.png'),cv2.imread(folder_path + '/plot_top_down.png')])
    left_image = cv2.resize(cv2.imread(folder_path + '/plot_normal.png'), (0,0), fx=2., fy=2.)
    cv2.imwrite(folder_path + '/all_views.png',cv2.hconcat([left_image,right_images]))

def plot_no_fly(ax,corners_no_fly_zone):
    x_min,y_min,z_min = corners_no_fly_zone[0,0].item(),corners_no_fly_zone[0,1].item(),corners_no_fly_zone[0,2].item()
    x_max,y_max,z_max = corners_no_fly_zone[1,0].item(),corners_no_fly_zone[1,1].item(),corners_no_fly_zone[1,2].item()
    vertices_1 = np.array([[x_min,y_min,z_min],[x_max,y_min,z_min],[x_max,y_max,z_min],[x_min,y_max,z_min]])
    vertices_2 = np.array([[x_min,y_min,z_max],[x_max,y_min,z_max],[x_max,y_max,z_max],[x_min,y_max,z_max]])
    vertices_3 = np.array([[x_min,y_min,z_min],[x_min,y_min,z_max],[x_max,y_min,z_max],[x_max,y_min,z_min]])
    vertices_4 = np.array([[x_min,y_max,z_min],[x_min,y_max,z_max],[x_max,y_max,z_max],[x_max,y_max,z_min]])
    vertices_5 = np.array([[x_min,y_min,z_min],[x_min,y_max,z_min],[x_min,y_max,z_max],[x_min,y_min,z_max]])
    vertices_6 = np.array([[x_max,y_min,z_min],[x_max,y_max,z_min],[x_max,y_max,z_max],[x_max,y_min,z_max]])
    list_vertices = [vertices_1,vertices_2,vertices_3,vertices_4,vertices_5,vertices_6]
    faces = Poly3DCollection(list_vertices, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    ax.add_collection3d(faces)

    return ax

def perform_test(network,max_moves,config,dataset,sampler,converter):

    all_images = torch.empty(max_moves,2,3,100,100).cuda()
    all_pos = torch.empty(max_moves,2*config["data"]["dim_position"]).cuda()
    all_predictions = torch.empty(max_moves,config["model"]["number_outputs"]).cuda()
    all_indices = torch.empty(max_moves,2).cuda()
    all_action_penalties = torch.empty(max_moves,config["model"]["number_outputs"]).cuda()

    history_dict = initialise_history_dict()
    # Randomly choose target
    indices = sampler.sample_allowed_pairs(1)[0]
    #indices = torch.tensor([40337,10902])
    pos_index = indices[0].item()
    target_index = indices[1].item()

    images,positions,_, action_penalties= dataset[indices]
    current_pos = positions[:config["data"]["dim_position"]]
    target_pos = positions[config["data"]["dim_position"]:2*config["data"]["dim_position"]]

    history_dict['target'] = list(target_pos.cpu().numpy())

    # Determine optimal number of steps
    steps_ideal_trajectory = nx.shortest_path_length(dataset.graph,pos_index, target_index + config["data"]["number_images"])
    
    counter = 0
    # Repeat until network says terminate or until have reached max_moves
    while counter < max_moves:
        # Find next move
        output,move = find_next_move(images, pos_index, current_pos.clone(),network,history_dict,converter)
        all_indices[counter] = indices
        all_images[counter] = images
        all_pos[counter] = positions.cpu()
        all_predictions[counter] = output
        all_action_penalties[counter] = action_penalties
        # Update history, position and position index
        counter +=1
        update_history_dict(pos_index,current_pos.cpu(),output,move,history_dict)
        if move == config["model"]["number_outputs"] - 1:
            history_dict['terminated'] = True
            break

        current_pos += converter.move_to_coords[move]
        current_pos = converter.map_pose(current_pos)

        # Check again that position is really allowed
        if  not converter.validate_pose(current_pos):
            raise Exception("Outside of valid space")

        pos_index = converter.pose_to_index(current_pos)

        #Get new images
        images,positions,indices,action_penalties= dataset[torch.tensor([pos_index, target_index])]
        indices = indices.cpu()
        action_penalties = action_penalties.cpu()

    # Return statistics depending on whether have reached target or not
    history_dict['statistics']['minimum_steps'] = steps_ideal_trajectory
    history_dict['statistics']['steps_took'] = counter
    history_dict['statistics']['reached_target'] = ((torch.abs(current_pos - target_pos) < 0.01).all().item() and history_dict['terminated'] == True )
    history_dict['trajectory'] = np.round(np.array(history_dict['trajectory']),1)
    history_dict['predictions'] = np.array(history_dict['predictions'])
    return history_dict, all_indices[:counter], all_images[:counter], all_pos[:counter], all_predictions[:counter], all_action_penalties[:counter]


def main(argv):
    del(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    exp_path = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/' + FLAGS.name
    config = load_config(exp_path +'/config.json')

    #load dataset
    dataset = load_data_set(config)

    sampler = Sampler(config)
    converter = Converter(config["data"]["min_pos"],config["data"]["max_pos"],config["data"]["steps"],config["data"]["number_images"],config["data"]["no_fly_zone"])
    
    
    #For testing obstruction load special dataset and modify some settings
    # normalise = torchvision.transforms.Normalize(mean=[0.4381, 0.4192, 0.3965],std=[0.0929, 0.0905, 0.0902])
    # dataset = Dataset('/data/cvfs/fml35/own_datasets/grid_world/obstruction_cube','positions.csv',config, transform = normalise)
    # dataset.config["data"]["occlusion_probability"] = 0.0
    #dataset.config["data"]["no_fly"] == True
    #converter.corners_no_fly_zone = torch.tensor([[-0.6,-4.2,0.7],[0.,-4.2,1.3]])
    
    #load model
    net = load_model(config)

    net.load_state_dict(torch.load(exp_path + '/checkpoints/' + FLAGS.weights))
    net.eval()
    net.cuda()

    # number of moves after which trial is stopped
    max_moves = 70

    dt_string = datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y")
    with torch.no_grad():
        for j in range(FLAGS.number_tests):
            history_dict, all_indices, all_images, all_pos, all_predictions, all_action_penalties = perform_test(net,max_moves,config,dataset,sampler,converter)
            print(history_dict)
            folder_path = '{}/visualisations/trajectories/{}_{}'.format(exp_path,dt_string,j)
            os.mkdir(folder_path)
            dict_file = folder_path + '/history_dict'
            write_to_file(dict_file + '.txt', str(history_dict))
            np.save(dict_file+'.npy', history_dict)
            plot_trajectory(dict_file +'.npy', folder_path,config,converter)
            visualise_pairs(all_images, all_pos, all_predictions,all_action_penalties ,folder_path,config, kind='trajectory')

if __name__ == "__main__":
    app.run(main)

