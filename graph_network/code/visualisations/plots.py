import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os 
import numpy as np
import torch
import pickle


def plot_history(file_path,store_path,epoch):
    df = pd.read_csv(file_path)
    metrics = ['loss','accuracy']

    min_epoch = 0
    if epoch > 50:
        min_epoch = 40

    for metric in metrics:
        fig = plt.figure()
        y = df['training_'+metric]
        plt.plot(df['epoch'][min_epoch:],y[min_epoch:], label= metric)
        plt.xlabel('Epochs')
        plt.legend()
        fig.savefig('{}/history_{}_epoch_{}.png'.format(store_path,metric,epoch),dpi=70)
        with open('{}/history_{}.pkl'.format(store_path,metric),'wb') as file:
            pickle.dump(fig, file)
        plt.close(fig)
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss', color='red')
    ax1.plot(df['epoch'][min_epoch:],df['training_loss'][min_epoch:], color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy', color='blue')
    ax2.plot(df['epoch'][min_epoch:],df['training_accuracy'][min_epoch:], color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()
    fig.savefig('{}/combined_epoch_{}.png'.format(store_path,epoch),dpi=70)
    plt.close(fig)



def visualise_steps_predictions(positions,predictions,targets,path, config, losses=None, kind = 'random_pairs'):

    positions = positions.cpu().numpy().round(4)
    predictions = predictions.cpu().numpy()
    if kind == 'random_pairs':
        number_visualisations = config["visualisations"]["random_pairs"]
    elif kind == 'hard_pairs':
        number_visualisations = config["visualisations"]["hard_pairs"]
    elif kind == 'trajectory':
        number_visualisations = positions.shape[0]
    for i in range(number_visualisations):
        fig = plt.figure(figsize=plt.figaspect(0.25))
   
        plt.figtext(0.05, 0.8, 'Position: {}'.format(list(positions[i,:4])), wrap=True, fontsize=12)
        plt.figtext(0.05, 0.72, 'Goal: {}'.format(list(positions[i,4:])), wrap=True, fontsize=12)
        plt.figtext(0.05, 0.6, 'True number of steps: {}'.format(targets[i].item()), wrap=True, fontsize=12)
        plt.figtext(0.05, 0.52, 'Predicted number of steps: {}'.format(predictions[i].item()), wrap=True, fontsize=12)
        if losses is not None:
            plt.figtext(0.05, 0.4, 'Loss: {:.4f}'.format(losses[i].item()), wrap=True, fontsize=12)
        
        ax = fig.add_subplot(1,4,2, projection='3d')
        ax = plot_position_and_target(ax,positions[i],config)
        counter = 3
        for angles in [[0.,270],[90.,270]]:
            ax = fig.add_subplot(1,4,counter, projection='3d')
            ax = plot_position_and_target(ax,positions[i],config)
            ax.view_init(elev=angles[0], azim=angles[1])
            counter += 1
        
        fig.savefig(path + '/pair_{}.png'.format(i))
        plt.close(fig)


def plot_position_and_target(ax,position,config):

    x_t,y_t,z_t = position[4],position[5],position[6]
    x_s,y_s,z_s = position[0],position[1],position[2]

    ax.scatter(x_t,y_t,z_t,label='Goal',color="green")
    ax.plot([x_t,x_t],[y_t,y_t],[0.1,z_t],'--',color='green')
    ax.scatter(x_s,y_s,z_s,label='Position',color='orange')
    ax.plot([x_s,x_s],[y_s,y_s],[0.1,z_s],'--',color='orange')
    # curent position
    dx_s, dy_s = - np.sin(2 * np.pi * position[3]), np.cos(2 * np.pi * position[3])
    ax.quiver(x_s,y_s,z_s, dx_s, dy_s, 0, length=0.4, color="orange",arrow_length_ratio=0.6)
    # target
    dx_t, dy_t = - np.sin(2 * np.pi * position[7]), np.cos(2 * np.pi * position[7])
    ax.quiver(x_t,y_t,z_t, dx_t, dy_t, 0, length=0.4, color="green",arrow_length_ratio=0.6)
    ax.legend()

    ax.set_xlabel('x - windows')
    ax.set_ylabel('y - kitchen')
    ax.set_zlabel('z')
    ax.set_xlim(-1.3,1.8)
    ax.set_ylim(-0.5,1.4)
    ax.set_zlim(0.2,1.7)
    if config["graph"]["no_fly"]:
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
