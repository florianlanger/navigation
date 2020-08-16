import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os 
import numpy as np
import torch
import pickle


def plot_comparison():
    name_plot = 'loss'
    name_exp1 = 'exp_14_1000_examples_time_10_04_11_date_15_04_2020'
    #name_exp2 = 'exp_11_overfit'

    file_path_to_csv_1 = '/../history.csv'.format(name_exp1)
    #file_path_to_csv_2 = '/../experiments/{}/history.csv'.format(name_exp2)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    df1 = pd.read_csv(dir_path + file_path_to_csv_1)
    #df2 = pd.read_csv(dir_path + file_path_to_csv_2)
    kinds = ['training','validation']
    metrics = ['loss','L2_distance','angle_difference']

    metric = 'loss'
    fig = plt.figure()
    for exp,df in zip([name_exp1],[df1]):
        for kind in kinds:
            plt.plot(df['epoch'],df[kind+'_'+metric], label=exp +'  ' + kind +' ' + metric)
    plt.xlabel('Epochs')
    plt.ylim(0,1)
    plt.legend()
    fig.savefig(dir_path+'/../visualisations/history/{}.png'.format(name_plot),dpi=70)
    plt.close(fig)

def plot_loss_vs_steps():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    loss = np.load(dir_path + '/../visualisations/testing/loss_vs_steps.npy')
    fig = plt.figure()

    fig, ax = plt.subplots()
    x = np.arange(0,46)
    y = loss[:,0]
    ax.scatter(x,y)

    ax.set_xlabel('Number of steps between position and goal')
    ax.set_ylim(0,0.22)
    ax.set_xlim(0,25)
    ax.set_ylabel('Loss (Random Guessing is 0.1, worst possible is 0.2)')
    for i, frequency in enumerate(loss[:,1]):
        if i < 10:
            ax.annotate(frequency, (x[i],y[i]+0.01))
        else:
            ax.annotate(frequency, (x[i],y[i]+0.01 + (i % 2)*0.03 ))
    fig.savefig(dir_path + '/../visualisations/testing/loss_vs_steps.png',dpi=70)
    plt.close(fig)

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



def plot_frequencies(counters_actions, epoch, exp_path, config):
    predictions = (counters_actions[0]/torch.sum(counters_actions[0])).cpu().numpy()
    ground_truth = (counters_actions[1]/torch.sum(counters_actions[1])).cpu().numpy()

    labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Rot +','Rot -','Terminate']
    x = np.arange(10)
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/2, predictions, width, label='Prediction')
    ax.bar(x + width/2, ground_truth, width, label='Ground truth')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of action being predicted/correct')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.savefig(exp_path + '/visualisations/frequencies/epoch_{}.png'.format(epoch))
    plt.close(fig)


def visualise_action_predictions(positions,targets,predictions,path, config, losses=None, kind = 'random_pairs'):

    positions = positions.cpu().numpy().round(2)
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    if kind == 'random_pairs':
        number_visualisations = config["visualisations"]["random_pairs"]
    elif kind == 'hard_pairs':
        number_visualisations = config["visualisations"]["hard_pairs"]
    elif kind == 'trajectory':
        number_visualisations = positions.shape[0]
    for i in range(number_visualisations):
        fig = plt.figure(figsize=plt.figaspect(0.2))
    
        labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Rot +','Rot -','Terminate']
        x = np.arange(10)
        width = 0.35  # the width of the bars

        ax = fig.add_subplot(1,4,1)
        ax.bar(x - width/2, predictions[i], width, label='Prediction')
        ax.bar(x + width/2, targets[i], width, label='Ground truth')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(prop={'size': 6})
        plt.setp(ax.get_xticklabels(), fontsize=6)
        plt.figtext(0.32, 0.95, 'Position: {}'.format(list(positions[i,:4])), wrap=True, fontsize=12)
        plt.figtext(0.32, 0.87, 'Goal: {}'.format(list(positions[i,4:])), wrap=True, fontsize=12)
        if losses is not None:
            plt.figtext(0.25, 0.8, 'loss: {:.4f}'.format(losses.cpu().numpy()[i]), wrap=True, fontsize=12)
        
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


def plot_heatmap_accuracies(path,epoch,accuracies,config):
    os.mkdir(path +'/epoch_{}'.format(epoch))

    x_min,x_max = config["data"]["min_pos"][0].item(),config["data"]["max_pos"][0].item()
    y_min, y_max = config["data"]["min_pos"][1].item(),config["data"]["max_pos"][1].item()
    zs = np.linspace(config["data"]["min_pos"][2].item(),config["data"]["max_pos"][2].item(),config["data"]["states_each_dimension"][2]).round(2)

    for i,kind in enumerate(['position','target']):
        for j,z in enumerate(zs):
            for k, angle in enumerate([0.,90.,180.,270.]):
                fig = plt.figure()
                plt.imshow(accuracies[i,:,:,j,k,1].t(), cmap='viridis', interpolation='none', extent=[x_min,x_max + 0.1,y_min,y_max + 0.1], origin='lower')
                plt.colorbar()
                current_cmap = matplotlib.cm.get_cmap()
                current_cmap.set_bad(color='white')
                plt.xticks()
                plt.yticks()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Accuracy')
                fig.savefig('{}/epoch_{}/{}_{}_{}.png'.format(path,epoch,kind,z,angle))
                plt.close(fig)

def visualise_image(image,path,text=None):

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    image1 = image.numpy()
    plt.imshow(np.moveaxis(np.clip(image1,0,1), 0, -1))
    if text is not None:
        plt.figtext(0.75, 0.5, text, wrap=True, horizontalalignment='center', fontsize=12)
    fig.savefig(path)
    plt.close(fig)

def plot_position_and_target(ax,position,config):

    x_t,y_t,z_t = position[4],position[5],position[6]
    x_s,y_s,z_s = position[0],position[1],position[2]

    ax.scatter(x_t,y_t,z_t,label='Goal',color="green")
    ax.plot([x_t,x_t],[y_t,y_t],[0.1,z_t],'--',color='green')
    ax.scatter(x_s,y_s,z_s,label='Position',color='orange')
    ax.plot([x_s,x_s],[y_s,y_s],[0.1,z_s],'--',color='orange')
    # curent position
    dx_s, dy_s = np.cos(2 * np.pi * position[3]), np.sin(2 * np.pi * position[3])
    ax.quiver(x_s,y_s,z_s, dx_s, dy_s, 0, length=0.4, color="orange",arrow_length_ratio=0.6)
    # target
    dx_t, dy_t = np.cos(2 * np.pi * position[7]), np.sin(2 * np.pi * position[7])
    ax.quiver(x_t,y_t,z_t, dx_t, dy_t, 0, length=0.4, color="green",arrow_length_ratio=0.6)
    ax.legend()

    ax.set_xlabel('x - windows')
    ax.set_ylabel('y - kitchen')
    ax.set_zlabel('z')
    ax.set_xlim(-1.9,3.1)
    ax.set_ylim(-1.,2.2)
    ax.set_zlim(0.0,2.4)



    corners_no_fly_zone = torch.tensor([[[-1.3000,  0.5000,  0.0000],
         [-0.1000,  1.7000,  0.9200]],

        [[-2.0500, -0.2800,  1.3900],
         [-1.7700,  1.6200,  2.0900]],

        [[-2.0630,  1.7200,  0.0000],
         [-1.3170,  2.2000,  0.9200]],

        [[-0.9000,  0.5400,  2.1900],
         [-0.6000,  0.8000,  2.5900]],

        [[-1.5400, -0.2800,  0.0200],
         [-2.0200,  1.6200,  0.8200]],

        [[ 0.5500, -0.5000,  0.1000],
         [ 1.6500,  1.1000,  0.9000]],

        [[-1.2200, -0.5200, -0.0100],
         [-0.5200,  0.1200,  0.4100]],

        [[ 2.1200,  0.1900, -0.0200],
         [ 3.1600,  1.1700,  0.7400]],

        [[-1.0200,  2.1600,  0.0000],
         [-0.0200,  2.3000,  0.8000]],

         [[ 1.9600,  2.5700,  0.6600],
         [ 2.3600,  2.9500,  0.9600]]])

    for no_fly_zone in corners_no_fly_zone:
        ax = plot_no_fly(ax,no_fly_zone.numpy())

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
