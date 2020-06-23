from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
import os


def one_view_trajectory(ax,history_dict,folder_path,converter):


    for i in range(history_dict["target_counter"]+1):
        color = 'blue'
        if i == history_dict["target_counter"]:
            color = 'orange'
        x_t,y_t,z_t = history_dict['predicted_targets'][i,0],history_dict['predicted_targets'][i,1],history_dict['predicted_targets'][i,2]
        ax.scatter(x_t,y_t,z_t,color=color)
        dx_t, dy_t = dx_t, dy_t = - np.sin(2 * np.pi * history_dict['predicted_targets'][i,3]), np.cos(2 * np.pi * history_dict['predicted_targets'][i,3])
        ax.quiver(x_t,y_t,z_t, dx_t, dy_t, 0, length=0.1, color=color,arrow_length_ratio=0.6)


    for name,color in zip(['true_poses','predicted_poses'],['green','red']):
        x,y,z = history_dict[name][:history_dict["counter"]+1,0],history_dict[name][:history_dict["counter"]+1,1],history_dict[name][:history_dict["counter"]+1,2]
        ax.scatter(x,y,z,linewidth=1,color=color,label=name,s=2)
        if name == 'true_poses':
            for i in range(len(x)):
                ax.text(x[i],y[i],z[i],str(i+1))
                dx, dy = - np.sin(2 * np.pi * x[i]), np.cos(2 * np.pi * y[i])
                ax.quiver(x[i],y[i],z[i], dx, dy, 0, length=0.1, color=color,arrow_length_ratio=0.6)

    
    ax.legend()

    ax.set_xlabel('x - windows')
    ax.set_ylabel('y - kitchen')
    ax.set_zlabel('z')

    ax.set_xlim(converter.min_position[0].item(),converter.max_position[0].item())
    ax.set_ylim(converter.min_position[1].item(),converter.max_position[1].item())
    ax.set_zlim(converter.min_position[2].item(),converter.max_position[2].item())

    ax = plot_no_fly(ax,np.array([[0.5,-0.5,0.2],[1.7,1.1,0.9]]))
    ax = plot_no_fly(ax,np.array([[-1.3,0.5,0.2],[-0.1,1.4,1.1]]))

    return ax
    
def plot_trajectory(history_dict,folder_path,converter,label):
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1 = one_view_trajectory(ax1,history_dict,folder_path,converter)

    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax2 = one_view_trajectory(ax2,history_dict,folder_path,converter)
    ax2.view_init(elev=0, azim=270)

    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3 = one_view_trajectory(ax3,history_dict,folder_path,converter)
    ax3.view_init(elev=90, azim=270)
   
    fig.savefig('{}/trajectory_{}_{}.png'.format(folder_path,str(history_dict["target_counter"]).zfill(2),label),dpi=150)
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


def plot_action_predictions(history_dict,test_folder,config):

    all_action_predictions = history_dict['action_predictions']
    all_predicted_poses = history_dict['predicted_poses']
    all_targets = history_dict['predicted_targets']

    labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Rot +','Rot -','Terminate']
    x = np.arange(10)
    width = 0.35  # the width of the bars
    if history_dict['last_counter'] == 0:
        os.mkdir(test_folder + '/action_predictions')

    upper_limit = history_dict['counter']
    if history_dict['terminated'] == True:
        upper_limit = history_dict['counter'] + 1
    for i in range(history_dict['last_counter'],upper_limit):
        fig = plt.figure(figsize=plt.figaspect(0.2))
        ax = fig.add_subplot(1,4,1)
        ax.bar(x, all_action_predictions[i], width, label='Prediction')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(prop={'size': 6})
        plt.setp(ax.get_xticklabels(), fontsize=6)
        target = all_targets[int(history_dict["target_indices"][i])].round(4)
        plt.figtext(0.1, 0.95, 'Position: {}'.format(list(all_predicted_poses[i].round(3))), wrap=True, fontsize=12)
        plt.figtext(0.1, 0.90, 'Goal: {}'.format(list(target)), wrap=True, fontsize=12)

        ax = fig.add_subplot(1,4,2, projection='3d')
        ax = plot_position_and_target(ax,all_predicted_poses[i],target,config)
        counter = 3
        for angles in [[0.,270],[90.,270]]:
            ax = fig.add_subplot(1,4,counter, projection='3d')
            ax = plot_position_and_target(ax,all_predicted_poses[i],target,config)
            ax.view_init(elev=angles[0], azim=angles[1])
            counter += 1

        fig.savefig(test_folder + '/action_predictions/prediction_{}.png'.format(str(i+1).zfill(2)))
        plt.close(fig)



def plot_position_and_target(ax,position,target,config):

    x_t,y_t,z_t,angle_t = target[0],target[1],target[2],target[3]
    x_s,y_s,z_s,angle_s = position[0],position[1],position[2],position[3]

    ax.scatter(x_t,y_t,z_t,label='Goal',color="green")
    ax.plot([x_t,x_t],[y_t,y_t],[0.1,z_t],'--',color='green')
    ax.scatter(x_s,y_s,z_s,label='Position',color='orange')
    ax.plot([x_s,x_s],[y_s,y_s],[0.1,z_s],'--',color='orange')
    # curent position
    dx_s, dy_s = - np.sin(2 * np.pi * angle_s), np.cos(2 * np.pi * angle_s)
    ax.quiver(x_s,y_s,z_s, dx_s, dy_s, 0, length=0.4, color="orange",arrow_length_ratio=0.6)
    # target
    dx_t, dy_t = - np.sin(2 * np.pi * angle_t), np.cos(2 * np.pi * angle_t)
    ax.quiver(x_t,y_t,z_t, dx_t, dy_t, 0, length=0.4, color="green",arrow_length_ratio=0.6)
    ax.legend()

    ax.set_xlabel('x - windows')
    ax.set_ylabel('y - kitchen')
    ax.set_zlabel('z')
    ax.set_xlim(-1.3,1.8)
    ax.set_ylim(-0.5,1.4)
    ax.set_zlim(0.2,1.7)
    if config["no_fly"] == "True":
        ax = plot_no_fly(ax,np.array([[0.5,-0.5,0.2],[1.7,1.1,0.9]]))
        ax = plot_no_fly(ax,np.array([[-1.3,0.5,0.2],[-0.1,1.4,1.1]]))
    return ax


def visualise_poses(history_dict,test_folder):
    if history_dict['last_counter'] == 0:
        os.mkdir(test_folder + '/poses')

    if history_dict['last_counter'] == 0:
        lower_limit = 0
    else:
        lower_limit = history_dict['last_counter']+1
    for i in range(lower_limit,history_dict["counter"]+1):
        image = Image.open(test_folder +'/images/' + history_dict['image_names'][i])
        image = F.to_tensor(image)[:3,:,:]
        path = test_folder + '/poses/pose_{}.png'.format(str(i+1).zfill(2))
        text = 'Pose: {} \nPredicted: {}'.format(
            str(history_dict['true_poses'][i]),str(history_dict['predicted_poses'][i]))
        visualise_image(image,path,text)
    # Treat target extra
    image = Image.open(test_folder +'/images/' + history_dict['target_name'])
    image = F.to_tensor(image)[:3,:,:]
    path = test_folder + '/poses/pose_target.png'
    text = 'Pose: {} \nPredicted: {}'.format(
            str(history_dict['true_target']),str(history_dict['predicted_targets'][0]))
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
