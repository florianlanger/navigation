import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os 
import numpy as np
import torch
import pickle

from losses.basic_losses import calc_target_distribution

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



def plot_frequencies(counters_actions, epoch, exp_path, config):
    predictions = (counters_actions[0]/torch.sum(counters_actions[0])).cpu().numpy()
    ground_truth = (counters_actions[1]/torch.sum(counters_actions[1])).cpu().numpy()
    if config["model"]["number_outputs"] == 8:
        labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Terminate']
        x = np.arange(8)
    elif config["model"]["number_outputs"] == 10:
        labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Rot +','Rot -','Terminate']
        x = np.arange(10)
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects_1 = ax.bar(x - width/2, predictions, width, label='Prediction')
    rects_2 = ax.bar(x + width/2, ground_truth, width, label='Ground truth')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of action being predicted/correct')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.savefig(exp_path + '/visualisations/frequencies/epoch_{}.png'.format(epoch))
    plt.close(fig)


def visualise_pairs(images, positions, predictions,action_penalties,path, config, losses=None, type_of_loss=None, kind = 'random_pairs'):
    
    target_distribution = calc_target_distribution(action_penalties)

    images = images.cpu().numpy()
    positions = positions.cpu().numpy().round(2)
    predictions = predictions.cpu().numpy()
    target_distribution = target_distribution.cpu().numpy()
    if kind == 'random_pairs':
        number_visualisations = config["visualisations"]["random_pairs"]
    elif kind == 'hard_pairs':
        number_visualisations = config["visualisations"]["hard_pairs"]
    elif kind == 'trajectory':
        number_visualisations = images.shape[0]
    for i in range(number_visualisations):
        fig = plt.figure()
        fig.add_subplot(2,2,1)
        image1 = images[i][0]
        plt.imshow(np.moveaxis(np.clip(image1,0,1), 0, -1))
        plt.gca().set_axis_off()
        plt.title('Position {}'.format(positions[i,:int(positions.shape[1]/2)]))
        fig.add_subplot(2,2,2)
        image2 = images[i][1]
        plt.imshow(np.moveaxis(np.clip(image2,0,1), 0, -1))
        plt.gca().set_axis_off()
        plt.title('Goal {}'.format(positions[i,int(positions.shape[1]/2):]))

        if config["model"]["number_outputs"] == 8:
            labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Terminate']
            x = np.arange(8)
        elif config["model"]["number_outputs"] == 10:
            labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Rot +','Rot -','Terminate']
            x = np.arange(10)
        width = 0.35  # the width of the bars

        ax = fig.add_subplot(2,2,3)
        ax.bar(x - width/2, predictions[i], width, label='Prediction')
        ax.bar(x + width/2, target_distribution[i], width, label='Ground truth')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(prop={'size': 6})
        plt.setp(ax.get_xticklabels(), fontsize=6)
        if losses is not None:
            plt.figtext(0.7, 0.1, '{} loss: {:.4f}'.format(type_of_loss, losses.cpu().numpy()[i]), wrap=True, horizontalalignment='center', fontsize=12)
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

def show_images(images,path):

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    image1 = images[0].numpy()
    plt.imshow(np.moveaxis(np.clip(image1,0,1), 0, -1))
    fig.add_subplot(1,2,2)
    image2 = images[1].numpy()
    plt.imshow(np.moveaxis(np.clip(image2,0,1), 0, -1))
    fig.savefig(path)
    plt.close(fig)

# exp_name = 'exp_16_VGG_200_epochs_time_11_12_11_date_07_05_2020'
# file_path = '../experiments/' + exp_name + '/history.csv'
# store_path =  '../experiments/' + exp_name + '/visualisations/history/'
# df = pd.read_csv(file_path)
# metrics = ['loss','accuracy']
# for metric in metrics:
#     fig = plt.figure()
#     plt.plot(df['epoch'][30:],df['training_'+metric][30:], label= metric)
#     plt.xlabel('Epochs')
#     plt.legend()
#     fig.savefig('{}/history_{}_zoomed.png'.format(store_path,metric),dpi=70)
