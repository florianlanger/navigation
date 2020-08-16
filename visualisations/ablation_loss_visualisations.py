from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import pickle


def one_view_probabilities(ax,output,target):

    list_high_prob_indices = []
    output = output
    #visualise predictions
    positions = np.empty((len(output),3))
    for i in range(len(output)):
        index_1, index_2, index_3 = i // 81,  (i % 81)  // 9, i % 9
        indices = np.array([index_1,index_2,index_3])
        #position = cube[:3] + (indices - 4) * cube[3:6]/2
        position = np.array([0.,0.,0.]) + (indices - 4) *  np.array([1.,1.,1.]) /2
        positions[i] = position
        if output[i] > 0.9:
            list_high_prob_indices.append(i)

    positions = positions[output>0.01]
    output = output[output>0.01]
    p = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=output,cmap='viridis_r')


    # # visualise target
    # for i in range(len(target)):
    #     if target[i] == 1:
    #         index_1, index_2, index_3 = i // 81,  (i % 81)  // 9, i % 9
    #         indices = np.array([index_1, index_2, index_3])
    #         #position = cube[:3] + (indices - 4) * cube[3:6]/2 + np.array([0.05,0.05,0.05])
    #         position = np.array([0.,0.,0.]) + (indices - 4) *  np.array([1.,1.,1.]) /2 + np.array([0.05,0.05,0.05])
    
    position = target + 0.15
    ax.scatter(position[0],position[1],position[2],color='red')    

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)

    # add cube 
    ax = plot_no_fly(ax,np.stack([np.array([-0.5,-0.5,-0.5]),np.array([0.5,0.5,0.5])]))

    return ax, p
    
def visualise(output,target,path):
    for i in range(output.shape[0]):
        fig = plt.figure(figsize=plt.figaspect(2.))
        ax1 = fig.add_subplot(2,1,1, projection='3d')
        ax1,p = one_view_probabilities(ax1,output[i],target[i])
        fig.colorbar(p, ax=ax1)
        ax1.view_init(elev=0, azim=180)

        ax2 = fig.add_subplot(2,1,2, projection='3d')
        ax2,_ = one_view_probabilities(ax2,output[i],target[i])
        ax2.view_init(elev=90, azim=180)
    
        fig.savefig(path + '_example_{}.png'.format(i),dpi=150,bbox_inches = "tight")
        plt.close(fig)


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
