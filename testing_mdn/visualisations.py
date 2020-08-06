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


def one_view_probability(ax,probability,folder_path,converter,config,anchor_object = None):
    points_per_dim = config["probability_grid"]["points_per_dim"]
    positions = np.empty([points_per_dim[0]*points_per_dim[1]*points_per_dim[2],3])
    probabilities = np.zeros(points_per_dim[0]*points_per_dim[1]*points_per_dim[2])
    counter = 0

    for indices, probability in np.ndenumerate(probability):
        if probability > 0.01 :
            positions[counter] = config["probability_grid"]["min_position"] + config["probability_grid"]["step_sizes"] * np.array(indices)
            probabilities[counter] = probability
        counter += 1 


    # positions = positions[probabilities>0.01]
    # probabilities = probabilities[probabilities>0.01]

    max_30_indices = np.argpartition(probabilities, -30)[-30:]
    positions = positions[max_30_indices]
    probabilities = probabilities[max_30_indices]

    position_max_prob = positions[np.argmax(probabilities)] 


    p = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=probabilities,norm=plt.Normalize(0, len(config[config["room"]]["scenarios_commands"][str(config['scenario'])])),cmap='viridis_r')

    # plot target position
    print(config[config["room"]]["scenarios_target_positions"])
    target_position = config[config["room"]]["scenarios_target_positions"][str(config["scenario"])]
    ax.scatter(target_position[0],target_position[1],target_position[2],color='black')

    # color max probability prediction red
    position_max_prob = positions[np.argmax(probabilities)] 
    ax.scatter(position_max_prob[0],position_max_prob[1],position_max_prob[2],color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y - max prob:'+str(np.max(probabilities).round(2)))
    ax.set_zlabel('z')

    dims = config[config["room"]]["dimensions"]
    ax.set_xlim(dims["min"][0],dims["max"][0])
    ax.set_ylim(dims["min"][1],dims["max"][1])
    ax.set_zlim(dims["min"][2],dims["max"][2])

    for no_fly_zone in converter.corners_no_fly_zone:
        ax = plot_no_fly(ax,no_fly_zone.numpy())
    
    if anchor_object is not None:
        single_object = config[config["room"]]["objects"][anchor_object]
        print(single_object)
        boundaries = torch.zeros((2,3))
        boundaries[0] = torch.from_numpy(np.array(single_object['dimensions'][:3]) - 4 * np.array(single_object['dimensions'][3:6]) * single_object['scaling'])
        boundaries[1] = torch.from_numpy(np.array(single_object['dimensions'][:3]) + 4 * np.array(single_object['dimensions'][3:6]) * single_object['scaling'])
        ax = plot_no_fly(ax,boundaries,(1,0,0,0.1))
    return ax,p
    
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

def plot_trajectory_1(history_dict,folder_path,converter):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, projection='3d')
    ax1 = one_view_trajectory(ax1,history_dict,folder_path,converter)

    path = '{}/trajectories/trajectory_{}'.format(folder_path,str(history_dict["counter"]).zfill(2))
    fig.savefig(path + '_view_1.png',dpi=150)
    
    ax1.view_init(elev=0, azim=180)
    fig.savefig(path + '_view_2.png',dpi=150)

    ax1.view_init(elev=90, azim=180)
    fig.savefig(path + '_view_3.png',dpi=150)
    plt.close(fig)

    img = cv2.hconcat([cv2.imread(path + '_view_3.png'),cv2.hconcat([cv2.imread(path + '_view_1.png'),cv2.imread(path + '_view_2.png')])])
    cv2.imwrite(path + '.png', img)

def plot_global_probability(probability,folder_path,converter,config,counter,kind,text=None,anchor_object = None):
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1,p = one_view_probability(ax1,probability,folder_path,converter,config,anchor_object)
    fig.colorbar(p, ax=ax1)
    ax1.view_init(elev=35, azim=200)

    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax2,_ = one_view_probability(ax2,probability,folder_path,converter,config,anchor_object)
    ax2.view_init(elev=0, azim=180)

    ax3 = fig.add_subplot(1,3,3, projection='3d')
    ax3,_ = one_view_probability(ax3,probability,folder_path,converter,config,anchor_object)
    ax3.view_init(elev=90, azim=180)
    
    if kind == 'total':
        plt.figtext(0.1, 0.95, 'Total distribution', wrap=True, fontsize=12)

    elif kind == 'last':
        plt.figtext(0.1, 0.95, 'Last contribution: {}'.format(text), wrap=True, fontsize=12)
    fig.savefig('{}/probabilities/probability_{}_{}.png'.format(folder_path,kind,str(counter).zfill(2)),dpi=150)
    plt.close(fig)

def plot_combined(history_dict,folder_path):
    i = history_dict["counter"]
    trajectory = cv2.resize(cv2.imread('{}/trajectories/trajectory_{}.png'.format(folder_path,str(i).zfill(2))),(3000,1000))
    # BLENDER
    camera_view = cv2.resize(cv2.imread(folder_path +'/images/' + history_dict['image_names'][i]),(1000,1000))
    #camera_view = cv2.resize(cv2.imread('/Users/legend98/Desktop/demo.png'),(1000,1000))
    action_predictions = cv2.resize(cv2.imread('{}/action_predictions/prediction_{}.png'.format(folder_path,str(i).zfill(2))),(4000,1000))
    #action_predictions = cv2.resize(cv2.imread('/Users/legend98/Desktop/demo.png'),(4000,1000))
    probabilities_total = cv2.resize(cv2.imread('{}/probabilities/probability_total_{}.png'.format(folder_path,str(history_dict['target_counter']-1).zfill(2))),(4000,1000))
    probabilities_last = cv2.resize(cv2.imread('{}/probabilities/probability_last_{}.png'.format(folder_path,str(history_dict['target_counter']-1).zfill(2))),(4000,1000))

    top_images = cv2.hconcat([trajectory,camera_view])
    top_images = cv2.vconcat([top_images,action_predictions])
    all_images = cv2.vconcat([top_images,probabilities_total])
    all_images = cv2.vconcat([all_images,probabilities_last])
    #all_images = cv2.resize(cv2.imread(folder_path + '/plot_normal.png'), (0,0), fx=2., fy=2.)
    cv2.imwrite(folder_path + '/combined/combined_{}.png'.format(str(i).zfill(2)),all_images)


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


def plot_action_predictions(history_dict,test_folder,config,converter):

    all_action_predictions = history_dict['action_predictions']
    all_predicted_poses = history_dict['predicted_poses']
    all_targets = history_dict['predicted_targets']

    labels = ['Stay', 'Pos x', 'Neg x', 'Pos y','Neg y', 'Pos z','Neg z','Rot +','Rot -','Terminate']
    x = np.arange(10)
    width = 0.35  # the width of the bars

    i = history_dict["counter"]
    fig = plt.figure(figsize=plt.figaspect(0.2))
    ax = fig.add_subplot(1,4,1)
    ax.bar(x, all_action_predictions[i], width, label='Prediction')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(prop={'size': 6})
    plt.setp(ax.get_xticklabels(), fontsize=6)
    target = all_targets[int(history_dict["target_counter"]-1)].round(4)
    plt.figtext(0.1, 0.95, 'Position: {}'.format(list(all_predicted_poses[i].round(3))), wrap=True, fontsize=12)
    plt.figtext(0.1, 0.90, 'Goal: {}'.format(list(target)), wrap=True, fontsize=12)

    counter = 2
    for angles in [[35,200],[0.,180],[90.,180]]:
        ax = fig.add_subplot(1,4,counter, projection='3d')
        ax = plot_position_and_target(ax,all_predicted_poses[i],target,config,converter)
        ax.view_init(elev=angles[0], azim=angles[1])
        counter += 1

    fig.savefig(test_folder + '/action_predictions/prediction_{}.png'.format(str(i).zfill(2)))
    plt.close(fig)



def plot_position_and_target(ax,position,target,config,converter):

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

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    dims = config[config["room"]]["dimensions"]
    ax.set_xlim(dims["min"][0],dims["max"][0])
    ax.set_ylim(dims["min"][1],dims["max"][1])
    ax.set_zlim(dims["min"][2],dims["max"][2])
    for no_fly_zone in converter.corners_no_fly_zone:
        ax = plot_no_fly(ax,no_fly_zone.numpy())
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
    # Treat target extra

    # BLENDER
    # image = Image.open(test_folder +'/images/' + history_dict['target_name'])
    #image = Image.open('/Users/legend98/Google Drive/MPhil project/navigation/testing/tests/test_03_time_17_34_15_date_14_07_2020/images/render_-1_x_-0.6207_y_0.9451_z_1.4787_rz_0.2334.png')
    # image = F.to_tensor(image)[:3,:,:]
    # path = test_folder + '/poses/pose_target.png'
    # text = 'Pose: {} \nPredicted: {}'.format(
    #         str(history_dict['true_target']),str(history_dict['predicted_targets'][0]))
    # visualise_image(image,path,text)

def visualise_image(image,path,text=None):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    image = image.numpy()
    plt.imshow(np.moveaxis(np.clip(image,0,1), 0, -1))
    if text is not None:
        plt.figtext(0.75, 0.5, text, wrap=True, horizontalalignment='center', fontsize=12)
    fig.savefig(path)
    plt.close(fig)
