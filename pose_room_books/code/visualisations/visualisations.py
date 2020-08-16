import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import pandas as pd
import pickle
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as F


def global_transformation(pose):
    pose = pose.numpy()
    position = pose[:3]*1000
    angle = np.array([(pose[3]-0.37)%1 * 360])
    rotation = R.from_euler('z',320,degrees=True)
    new_position = 0.00217*rotation.apply(position) + [2.9,2.05,1.05]
    return np.concatenate((new_position,angle)).round(4)

def visualise_poses(batch_idx, epoch,images,outputs,targets,losses,L2_dist,angle_diff,config,exp_path,kind,image_names):
    if batch_idx == 0 and (epoch-1) % config["visualisations"]["interval"] == 0:
        dir_path = '{}/visualisations/poses/epoch_{}'.format(exp_path,epoch)
        if kind == 'train':
            os.mkdir(dir_path)
        images = images.cpu()
        outputs = outputs.cpu()
        targets = targets.cpu()
        for i in range(min(config["visualisations"]["number"],config["training"]["batch_size"])):
            predicted_pose = global_transformation(outputs[i,:])
            target_pose = global_transformation(targets[i,:])

            path = dir_path + '/{}_{}.png'.format(kind,i)
            text = 'Predicted: {} \nTarget: {}\nLoss: {:.4f}\n L2 dist: {:.4f}\nAngle Diff: {:.4f}°\n{}'.format(
                str(predicted_pose),str(target_pose),losses[i].item(),L2_dist[i].item(),angle_diff[i].item(),image_names[i])
            
            visualise_single_image(images[i],torch.from_numpy(predicted_pose).unsqueeze(0),path,torch.from_numpy(target_pose),text)


def visualise_single_image(image,predicted_pose,path,target_pose=None,text=None,):
    fig = plt.figure(figsize=plt.figaspect(0.25))
    ax1 = fig.add_subplot(1,5,1, projection='3d')
    ax1 = one_view_pose(ax1,predicted_pose,target_pose)
    ax1.view_init(elev=35, azim=200)

    ax2 = fig.add_subplot(1,5,2, projection='3d')
    ax2 = one_view_pose(ax2,predicted_pose,target_pose)
    ax2.view_init(elev=0, azim=180)

    ax3 = fig.add_subplot(1,5,3, projection='3d')
    ax3 = one_view_pose(ax3,predicted_pose,target_pose)
    ax3.view_init(elev=90, azim=180)

    ax4 = fig.add_subplot(1,5,4)
    image1 = image.numpy()
    plt.imshow(np.moveaxis(np.clip(image1,0,1), 0, -1))
   
    plt.figtext(0.85, 0.5, text, wrap=True, horizontalalignment='center', fontsize=12)

    fig.savefig(path,dpi=300)
    plt.close(fig)

def visualise_trajectory_callback(predicted_poses,path):
    fig = plt.figure(figsize=plt.figaspect(0.33))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1 = one_view_pose(ax1,predicted_poses)
    ax1.view_init(elev=35, azim=200)

    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax2 = one_view_pose(ax2,predicted_poses)
    ax2.view_init(elev=0, azim=180)

    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3 = one_view_pose(ax3,predicted_poses)
    ax3.view_init(elev=90, azim=180)

    fig.savefig(path,dpi=300)
    plt.close(fig)
    
    
def visualise_callbacks(network,exp_path,epoch):
    target_folder = exp_path + '/visualisations/callbacks/epoch_' + str(epoch).zfill(6)
    os.mkdir(target_folder)
    for number,name in zip([18,26,20,16],['back','front','left','right']): #zip([18,26,20,16],['back','front','left','right']):
        folder_images = exp_path + '/../../callbacks_images/' + name
        new_target_folder = target_folder+'/'+name
        os.mkdir(new_target_folder)
        visualise_single_folder_callback(network,folder_images,new_target_folder,number)

def visualise_single_folder_callback(network,folder_images,new_target_folder,number_images):
    
    all_poses = np.zeros((number_images,4))

    for i in range(number_images):
        image = Image.open(folder_images+ '/image_' + str(i+1).zfill(6) + '.png')
        image = image.resize((128,96))
        image = F.to_tensor(image)[:3,:,:].unsqueeze(0).cuda()
        output = network(image)[0].cpu().detach()
        pose = global_transformation(output)
        visualise_single_image(image.cpu()[0],torch.from_numpy(pose).unsqueeze(0),new_target_folder + '/image_' + str(i+1).zfill(6) + '.png')
        all_poses[i] = pose

    visualise_trajectory_callback(torch.from_numpy(all_poses),new_target_folder + '/a_trajectory.png')



def one_view_pose(ax,predicted_pose,target_pose=None):
    
    for i in range(predicted_pose.shape[0]):
        x,y,z,angle = predicted_pose[i,0].item(),predicted_pose[i,1].item(),predicted_pose[i,2].item(),predicted_pose[i,3].item()
        if i == 0:
            ax.scatter(x,y,z,linewidth=1,color="red",label='Prediction',s=2)
        else:
            ax.scatter(x,y,z,linewidth=1,color="red",s=2)
        dx, dy = np.cos(2 * np.pi * angle/360), np.sin(2 * np.pi * angle/360)
        ax.quiver(x,y,z, dx, dy, 0, length=0.2, color='pink')
        ax.text(x,y,z,str(i+1))

    if target_pose is not None:
        x,y,z,angle = target_pose[0].item(),target_pose[1].item(),target_pose[2].item(),target_pose[3].item()
        ax.scatter(x,y,z,linewidth=1,color="green",label="Target",s=2)
        dx, dy = np.cos(2 * np.pi * angle/360), np.sin(2 * np.pi * angle/360)
        ax.quiver(x,y,z, dx, dy, 0, length=0.2, color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    ax.set_xlim(0,4.7)
    ax.set_ylim(0,4.6)
    ax.set_zlim(0,1.8)

    corners_no_fly_zone = torch.tensor([[[3.5000, 0.0000, 0.0000],
         [4.7000, 0.5800, 1.1800]],

        [[1.6800, 1.2000, 0.0000],
         [2.5800, 3.2400, 1.0000]],

        [[2.5800, 1.4000, 0.0000],
         [4.5800, 3.2600, 0.6200]],

        [[3.0800, 4.1800, 0.0000],
         [4.1800, 4.6800, 0.7800]],

        [[4.4400, 3.5400, 0.4600],
         [4.6400, 3.7400, 0.9400]],

        [[4.3200, 3.4000, 0.0000],
         [4.7200, 3.8000, 0.4600]],

        [[3.3000, 4.5000, 0.7800],
         [3.9400, 4.7400, 1.4000]],

        [[0.0000, 1.9000, 0.0000],
         [0.5400, 2.8600, 0.3800]]])


    for no_fly_zone in corners_no_fly_zone:
        ax = plot_no_fly(ax,no_fly_zone.numpy())

    return ax


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

