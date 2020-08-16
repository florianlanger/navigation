from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import pickle

#color=viridis(output[i].item())
def one_view_probabilities(ax,output,target,cube,text=None):

    list_high_prob_indices = []
    output = output.numpy()
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


    # visualise target
    for i in range(len(target)):
        if target[i] == 1:
            index_1, index_2, index_3 = i // 81,  (i % 81)  // 9, i % 9
            indices = np.array([index_1, index_2, index_3])
            #position = cube[:3] + (indices - 4) * cube[3:6]/2 + np.array([0.05,0.05,0.05])
            position = np.array([0.,0.,0.]) + (indices - 4) *  np.array([1.,1.,1.]) /2 + np.array([0.05,0.05,0.05])
            ax.scatter(position[0],position[1],position[2],color='red')    


    if text:
        plt.figtext(0.3, 0.9, text, wrap=True, horizontalalignment='center', fontsize=12)

    plt.figtext(0.7, 0.9, str(list_high_prob_indices), wrap=True, horizontalalignment='center', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)

    # add cube 
    ax = plot_no_fly(ax,np.stack([np.array([-0.5,-0.5,-0.5]),np.array([0.5,0.5,0.5])]))

    return ax, p
    
def visualise(output,target,cube,description,path,config):
    for i in range(config["visualisations"]["number"]):
        fig = plt.figure(figsize=plt.figaspect(0.3))
        ax1 = fig.add_subplot(1,3,1, projection='3d')
        ax1,p = one_view_probabilities(ax1,output[i],target[i],cube[i],description[i])
        fig.colorbar(p, ax=ax1)
        ax1.view_init(elev=40, azim=200)

        ax2 = fig.add_subplot(1,3,2, projection='3d')
        ax2,_ = one_view_probabilities(ax2,output[i],target[i],cube[i])
        ax2.view_init(elev=0, azim=180)

        ax3 = fig.add_subplot(1,3,3, projection='3d')
        ax3,_ = one_view_probabilities(ax3,output[i],target[i],cube[i])
        ax3.view_init(elev=90, azim=180)
    
        fig.savefig(path + '_example_{}.png'.format(i),dpi=150)
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

def plot_history(file_path,store_path,epoch):
    df = pd.read_csv(file_path)
    path = '{}/epoch_{}'.format(store_path,epoch)
    os.mkdir(path)
    metrics = ['loss','acc']
    min_epoch = 0
    if epoch > 50:
        min_epoch = 40

    for metric in metrics:
        plot_one_metric(metric,['train','val'],df,min_epoch,path)
    plot_combined(['loss','acc'],[['train','val'],['train','val']],df,min_epoch,path)

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