import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle

def visualise_poses(batch_idx, epoch,images,outputs,targets,losses,L2_dist,angle_diff,config,exp_path,kind):
    if batch_idx == 0 and (epoch-1) % config["visualisations"]["interval"] == 0:
        dir_path = '{}/visualisations/poses/epoch_{}'.format(exp_path,epoch)
        if kind == 'train':
            os.mkdir(dir_path)
        images = images.cpu()
        for i in range(min(config["visualisations"]["number"],config["training"]["batch_size"])):
            path = dir_path + '/{}_{}.png'.format(kind,i)
            text = 'Pose: {} \nPredicted: {}\nLoss: {:.4f}\n L2 dist: {:.4f}\nAngle Diff: {:.4f}°'.format(
                str(targets[i].cpu().numpy().round(4)),str(outputs[i].cpu().numpy().round(4)),losses[i].item(),L2_dist[i].item(),angle_diff[i].item())
            visualise_image(images[i],path,text)

def visualise_image(image,path,text=None):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    image1 = image.numpy()
    plt.imshow(np.moveaxis(np.clip(image1,0,1), 0, -1))
    if text is not None:
        plt.figtext(0.75, 0.5, text, wrap=True, horizontalalignment='center', fontsize=12)
    fig.savefig(path)
    plt.close(fig)

def plot_history(file_path,store_path,epoch):
    df = pd.read_csv(file_path)
    path = '{}/epoch_{}'.format(store_path,epoch)
    os.mkdir(path)
    metrics = ['loss','L2_dist','angle_difference']
    min_epoch = 0
    if epoch > 50:
        min_epoch = 40

    for metric in metrics:
        plot_one_metric(metric,['train','val'],df,min_epoch,path)
    plot_combined(['loss','L2_dist'],[['train','val'],['train','val']],df,min_epoch,path)

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

