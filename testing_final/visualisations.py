from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import os
import cv2


def one_view_trajectory(ax,history_dict,folder_path,converter):
    
    for i in range(history_dict["target_counter"]):
        label = 'Target {}'.format(i)
        if i == history_dict["target_counter"]-1:
            label = "Current Target"
        x_t,y_t,z_t = history_dict['predicted_targets'][i,0],history_dict['predicted_targets'][i,1],history_dict['predicted_targets'][i,2]
        ax.scatter(x_t,y_t,z_t,label=label)
        dx_t, dy_t = dx_t, dy_t = - np.sin(2 * np.pi * history_dict['predicted_targets'][i,3]), np.cos(2 * np.pi * history_dict['predicted_targets'][i,3])
        ax.quiver(x_t,y_t,z_t, dx_t, dy_t, 0, length=0.1,arrow_length_ratio=0.6)


    for name,color in zip(['predicted_poses'],['red']):
        x,y,z,angle = history_dict[name][:history_dict["counter"]+1,0],history_dict[name][:history_dict["counter"]+1,1],history_dict[name][:history_dict["counter"]+1,2],history_dict[name][:history_dict["counter"]+1,3]
        ax.scatter(x,y,z,linewidth=1,color=color,label=name,s=2)
        for i in range(len(x)):
            ax.text(x[i],y[i],z[i],str(i+1))
            dx, dy = np.cos(2 * np.pi * angle[i]/360), np.sin(2 * np.pi * angle[i]/360)
            ax.quiver(x[i],y[i],z[i], dx, dy, 0, length=0.2, color='green'),#arrow_length_ratio=0.6)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(converter.min_position[0].item(),converter.max_position[0].item())
    ax.set_ylim(converter.min_position[1].item(),converter.max_position[1].item())
    ax.set_zlim(converter.min_position[2].item(),converter.max_position[2].item())

    for no_fly_zone in converter.corners_no_fly_zone:
        ax = plot_no_fly(ax,no_fly_zone.numpy())

    return ax

def plot_trajectory(history_dict,folder_path,converter):
    fig = plt.figure(figsize=plt.figaspect(0.25))
    ax1 = fig.add_subplot(1,4,1, projection='3d')
    ax1 = one_view_trajectory(ax1,history_dict,folder_path,converter)
    ax1.view_init(elev=35, azim=200)

    ax2 = fig.add_subplot(1,4,2, projection='3d')
    ax2 = one_view_trajectory(ax2,history_dict,folder_path,converter)
    ax2.view_init(elev=0, azim=180)

    ax3 = fig.add_subplot(1,4,3, projection='3d')
    ax3 = one_view_trajectory(ax3,history_dict,folder_path,converter)
    ax3.view_init(elev=90, azim=180)

    ax4 = fig.add_subplot(1,4,4)
    ax4 = visualise_pose(ax4,history_dict,folder_path)
   
    text = 'Predicted: {} Action: {}'.format(str(history_dict['predicted_poses'][history_dict["counter"]]),history_dict['actions'][history_dict["counter"]])
    plt.figtext(0.5, 0.03, text, wrap=True, horizontalalignment='center', fontsize=12)

    fig.savefig('{}/trajectories/trajectory_{}.png'.format(folder_path,str(history_dict["counter"]).zfill(2)),dpi=300)
    plt.close(fig)


def one_view_probability(ax,probability,folder_path,converter,config,anchor_object = None):
    points_per_dim = config["probability_grid"]["points_per_dim"]
    positions = np.empty([points_per_dim[0]*points_per_dim[1]*points_per_dim[2],3])
    probabilities = np.zeros(points_per_dim[0]*points_per_dim[1]*points_per_dim[2])
    counter = 0

    for indices, probability in np.ndenumerate(probability):
        if probability > 0.01 :
            positions[counter] = config["probability_grid"]["min_position"] + config["probability_grid"]["step_sizes"] * np.array(indices)
            probabilities[counter] = probability
        counter += 1 


    # positions = positions[probabilities>0.01]
    # probabilities = probabilities[probabilities>0.01]

    max_30_indices = np.argpartition(probabilities, -30)[-30:]
    positions = positions[max_30_indices]
    probabilities = probabilities[max_30_indices]

    position_max_prob = positions[np.argmax(probabilities)] 


    p = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=probabilities,norm=plt.Normalize(0,1),cmap='viridis_r')


    # color max probability prediction red
    position_max_prob = positions[np.argmax(probabilities)] 
    ax.scatter(position_max_prob[0],position_max_prob[1],position_max_prob[2],color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y - max prob:'+str(np.max(probabilities).round(2)))
    ax.set_zlabel('z')

    dims = config[config["room"]]["dimensions"]
    ax.set_xlim(dims["min"][0],dims["max"][0])
    ax.set_ylim(dims["min"][1],dims["max"][1])
    ax.set_zlim(dims["min"][2],dims["max"][2])

    for no_fly_zone in converter.corners_no_fly_zone:
        ax = plot_no_fly(ax,no_fly_zone.numpy())
    
    if anchor_object is not None:
        single_object = config[config["room"]]["objects"][anchor_object]
        print(single_object)
        boundaries = torch.zeros((2,3))
        boundaries[0] = torch.from_numpy(np.array(single_object['dimensions'][:3]) - 4 * np.array(single_object['dimensions'][3:6]) * single_object['scaling'])
        boundaries[1] = torch.from_numpy(np.array(single_object['dimensions'][:3]) + 4 * np.array(single_object['dimensions'][3:6]) * single_object['scaling'])
        ax = plot_no_fly(ax,boundaries,(1,0,0,0.1))
    return ax,p

def plot_global_probability(probability,folder_path,converter,config,counter,kind,text=None,anchor_object = None):
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1,p = one_view_probability(ax1,probability,folder_path,converter,config,anchor_object)
    fig.colorbar(p, ax=ax1)
    ax1.view_init(elev=35, azim=200)

    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax2,_ = one_view_probability(ax2,probability,folder_path,converter,config,anchor_object)
    ax2.view_init(elev=0, azim=180)

    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3,_ = one_view_probability(ax3,probability,folder_path,converter,config,anchor_object)
    ax3.view_init(elev=90, azim=180)
    
    if kind == 'total':
        plt.figtext(0.1, 0.95, 'Total distribution', wrap=True, fontsize=12)

    elif kind == 'last':
        plt.figtext(0.1, 0.95, 'Last contribution: {}'.format(text), wrap=True, fontsize=12)
    fig.savefig('{}/probabilities/probability_{}_{}.png'.format(folder_path,kind,str(counter).zfill(2)),dpi=150)
    plt.close(fig)


def plot_combined(history_dict,folder_path):
    i = history_dict["counter"]
    trajectory = cv2.resize(cv2.imread('{}/trajectories/trajectory_{}.png'.format(folder_path,str(i).zfill(2))),(2000,500))

    probabilities_total = cv2.resize(cv2.imread('{}/probabilities/probability_total_{}.png'.format(folder_path,str(history_dict['target_counter']-1).zfill(2))),(2000,500))
    probabilities_last = cv2.resize(cv2.imread('{}/probabilities/probability_last_{}.png'.format(folder_path,str(history_dict['target_counter']-1).zfill(2))),(2000,500))

    all_images = cv2.vconcat([trajectory,probabilities_total])
    all_images = cv2.vconcat([all_images,probabilities_last])
    cv2.imwrite(folder_path + '/combined/combined_{}.png'.format(str(i).zfill(2)),all_images)


def plot_no_fly(ax,corners_no_fly_zone,color=(0,0,1,0.1)):
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
    faces.set_facecolor(color)

    ax.add_collection3d(faces)

    return ax

def visualise_pose(ax,history_dict,test_folder):

    image = Image.open(test_folder +'/images/' + history_dict['image_names'][history_dict["counter"]])
    image = F.to_tensor(image)[:3,:,:]
    image = image.numpy()
    ax.imshow(np.moveaxis(np.clip(image,0,1), 0, -1))
    return ax
