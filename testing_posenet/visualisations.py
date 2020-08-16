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


    for name,color in zip(['true_poses','predicted_poses'],['orange','red']):
        x,y,z = history_dict[name][:history_dict["counter"]+1,0],history_dict[name][:history_dict["counter"]+1,1],history_dict[name][:history_dict["counter"]+1,2]
        ax.scatter(x,y,z,linewidth=1,color=color,label=name,s=2)
        if name == 'true_poses':
            for i in range(len(x)):
                ax.text(x[i],y[i],z[i],str(i+1))
                dx, dy = - np.sin(2 * np.pi * x[i]), np.cos(2 * np.pi * y[i])
                ax.quiver(x[i],y[i],z[i], dx, dy, 0, length=0.1, color=color,arrow_length_ratio=0.6)

    
    ax.legend()

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
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1 = one_view_trajectory(ax1,history_dict,folder_path,converter)
    ax1.view_init(elev=35, azim=200)

    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax2 = one_view_trajectory(ax2,history_dict,folder_path,converter)
    ax2.view_init(elev=0, azim=180)

    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3 = one_view_trajectory(ax3,history_dict,folder_path,converter)
    ax3.view_init(elev=90, azim=180)
   
    fig.savefig('{}/trajectories/trajectory_{}.png'.format(folder_path,str(history_dict["counter"]).zfill(2)),dpi=150)
    plt.close(fig)


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

def visualise_poses(history_dict,test_folder):
    if history_dict['last_counter'] == 0:
        lower_limit = 0
    else:
        lower_limit = history_dict['last_counter']+1
    for i in range(lower_limit,history_dict["counter"]+1):
        # BLENDER
        image = Image.open(test_folder +'/images/' + history_dict['image_names'][i])
        #image = Image.open('/Users/legend98/Google Drive/MPhil project/navigation/testing/tests/test_03_time_17_34_15_date_14_07_2020/images/render_-1_x_-0.6207_y_0.9451_z_1.4787_rz_0.2334.png')
        image = F.to_tensor(image)[:3,:,:]
        path = test_folder + '/poses/pose_{}.png'.format(str(i+1).zfill(2))
        text = 'Pose: {} \nPredicted: {}'.format(
            str(history_dict['true_poses'][i]),str(history_dict['predicted_poses'][i]))
        visualise_image(image,path,text)

def visualise_image(image,path,text=None):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    image = image.numpy()
    plt.imshow(np.moveaxis(np.clip(image,0,1), 0, -1))
    if text is not None:
        plt.figtext(0.75, 0.5, text, wrap=True, horizontalalignment='center', fontsize=12)
    fig.savefig(path)
    plt.close(fig)