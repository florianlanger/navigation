import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os 
import numpy as np
import torch
import pickle

from losses.losses import pose_loss,L2_distance,angle_difference


def plot_history(file_path,store_path,epoch):
    df = pd.read_csv(file_path)
    path = '{}/epoch_{}'.format(store_path,epoch)
    os.mkdir(path)
    metrics = ['loss','L2_dist','angle_difference']
    min_epoch = 0
    if epoch > 50:
        min_epoch = 40

    for metric in metrics:
        plot_one_metric(metric,['train'],df,min_epoch,path)
    plot_combined(['loss','L2_dist'],[['train'],['train']],df,min_epoch,path)

def plot_one_metric(metric,kinds,df,min_epoch,path):
    fig = plt.figure()
    for kind in kinds:
        y = df[kind + '_' + metric]
        plt.plot(df['epoch'][min_epoch:],y[min_epoch:], label= kind + '_' + metric)
    plt.xlabel('Epochs')
    plt.legend()
    fig.savefig('{}/{}.png'.format(path,metric),dpi=70)
    with open('{}/{}.pkl'.format(path,metric),'wb') as file:
        pickle.dump(fig, file)
    plt.close(fig)

def plot_combined(metrics,kinds,df,min_epoch,path):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(metrics[0], color='red')
    for kind,style in zip(kinds[0],['solid','dashed']):
        ax1.plot(df['epoch'][min_epoch:],df[kind+'_'+metrics[0]][min_epoch:], color='red',linestyle=style,label=kind+'_'+metrics[0])
    ax1.tick_params(axis='y', labelcolor='red')
    plt.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.set_ylabel(metrics[1], color='blue')
    for kind,style in zip(kinds[1],['solid','dashed']):
        ax2.plot(df['epoch'][min_epoch:],df[kind+'_'+metrics[1]][min_epoch:], color='blue',linestyle=style,label=kind+'_'+metrics[1])
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.legend(loc='upper center')

    fig.tight_layout()
    fig.savefig('{}/combined_{}_{}.png'.format(path,metrics[0],metrics[1]),dpi=70)
    plt.close(fig)



def visualise_target_predictions(poses,instructions,target_poses,outputs,exp_path, config,epoch):


    path = exp_path + '/visualisations/single_pairs/epoch_{}'.format(epoch)
    os.mkdir(path)

    index_to_instruction = {}
    index_to_instruction[0] = 'Stay'
    index_to_instruction[1] = 'Go forward'
    index_to_instruction[2] = 'Go backward'
    index_to_instruction[3] = 'Go left'
    index_to_instruction[4] = 'Go right'
    index_to_instruction[5] = 'Go up'
    index_to_instruction[6] = 'Go down'
    index_to_instruction[7] = 'Rotate acw'
    index_to_instruction[8] = 'Rotate cw'


    for i in range(config["visualisations"]["random_pairs"]):
        fig = plt.figure(figsize=plt.figaspect(0.25))
   
        plt.figtext(0.05, 0.8, 'Position: {}'.format(list(poses[i].numpy().round(4))), wrap=True, fontsize=12)
        plt.figtext(0.05, 0.72, 'Instruction: {}'.format(index_to_instruction[(instructions[i]==1.).nonzero().item()], wrap=True, fontsize=16))

        plt.figtext(0.05, 0.6, 'True new target: {}'.format(list(target_poses[i].numpy().round(4))), wrap=True, fontsize=12)
        plt.figtext(0.05, 0.52, 'Predicted new target: {}'.format(list(outputs[i].numpy().round(4))), wrap=True, fontsize=12)

        loss = pose_loss(outputs[i].view(1,-1),target_poses[i].view(1,-1)).item()
        L2_dist,angle_diff = L2_distance(outputs[i,:3].view(1,-1),target_poses[i,:3].view(1,-1)).item(),angle_difference(outputs[i,3].view(1,-1),target_poses[i,3].view(1,-1)).item()
        plt.figtext(0.05, 0.4, 'Loss: {:.4f} L2 dist: {:.4f} Angle diff: {:.4f}'.format(loss, L2_dist,360 * angle_diff), wrap=True, fontsize=12)
        
        ax = fig.add_subplot(1,4,2, projection='3d')
        ax = plot_position_and_target(ax,[poses[i],target_poses[i],outputs[i]],config)
        counter = 3
        for angles in [[0.,270],[90.,270]]:
            ax = fig.add_subplot(1,4,counter, projection='3d')
            ax = plot_position_and_target(ax,[poses[i].numpy().round(4),target_poses[i].numpy().round(4),outputs[i].numpy().round(4)],config)
            ax.view_init(elev=angles[0], azim=angles[1])
            counter += 1
        
        fig.savefig(path + '/pair_{}.png'.format(i))
        plt.close(fig)


def plot_position_and_target(ax,list_poses,config):

    for pose,label,color in zip(list_poses,['Current pose','True Target Pose','Predicted Target Pose'],['blue','green','red']):
        x,y,z= pose[0],pose[1],pose[2]
        ax.scatter(x,y,z,label=label,color=color)
        ax.plot([x,x],[y,y],[0.1,z],'--',color=color)
        dx, dy = - np.sin(2 * np.pi * pose[3]), np.cos(2 * np.pi * pose[3])
        ax.quiver(x,y,z, dx, dy, 0, length=0.4, color=color,arrow_length_ratio=0.6)

    ax.legend()
    ax.set_xlabel('x - windows')
    ax.set_ylabel('y - kitchen')
    ax.set_zlabel('z')
    ax.set_xlim(-1.3,1.8)
    ax.set_ylim(-0.5,1.4)
    ax.set_zlim(0.2,1.7)
    ax = plot_no_fly(ax,np.array([[0.5,-0.5,0.2],[1.7,1.1,0.9]]))
    ax = plot_no_fly(ax,np.array([[-1.3,0.5,0.2],[-0.1,1.4,1.1]]))
    return ax


def plot_no_fly(ax,corners_no_fly_zone):
    x_min,y_min,z_min = corners_no_fly_zone[0,0],corners_no_fly_zone[0,1],corners_no_fly_zone[0,2]
    x_max,y_max,z_max = corners_no_fly_zone[1,0],corners_no_fly_zone[1,1],corners_no_fly_zone[1,2]
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
