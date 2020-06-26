import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import csv
import cv2


def plot(objects,current_pose,target_pose,dim_of_space,path,counter):

    fig = plt.figure(figsize=plt.figaspect(0.33))
    ax = fig.add_subplot(1,3,1, projection='3d')
    ax = plot_position_and_target(ax,objects,current_pose,target_pose,dim_of_space)
    j = 2

    for angles in [[0.,270],[90.,270]]:
        ax = fig.add_subplot(1,4,j, projection='3d')
        ax = plot_position_and_target(ax,objects,current_pose,target_pose,dim_of_space)
        ax.view_init(elev=angles[0], azim=angles[1])
        j += 1

    file_path = path + '/images/image_{}.png'.format(str(counter).zfill(5))
    fig.savefig(file_path)
    plt.close(fig)
    image = cv2.imread(file_path) 
    cv2.imshow('Image', image)  
    cv2.waitkey(0)  


def plot_position_and_target(ax,objects,current_pose,target_pose,dim_of_space):

    x_t,y_t,z_t,angle_t = target_pose[0],target_pose[1],target_pose[2],target_pose[3]
    x_s,y_s,z_s,angle_s = current_pose[0],current_pose[1],current_pose[2],current_pose[3]

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
    ax.set_xlim(dim_of_space[0],dim_of_space[3])
    ax.set_ylim(dim_of_space[1],dim_of_space[4])
    ax.set_zlim(dim_of_space[2],dim_of_space[5])

    colors = ['blue','red','green','yellow','magenta','chocolate','lightsteelblue']
    number_objects = len(objects)
    for i,key in enumerate(objects):
        ax = plot_no_fly(ax,objects[key],str(key),colors[i])
    return ax


def plot_no_fly(ax,corners_no_fly_zone,label,color):
    x_min,y_min,z_min = corners_no_fly_zone[0],corners_no_fly_zone[1],corners_no_fly_zone[2]
    x_max,y_max,z_max = corners_no_fly_zone[3],corners_no_fly_zone[4],corners_no_fly_zone[5]
    vertices_1 = np.array([[x_min,y_min,z_min],[x_max,y_min,z_min],[x_max,y_max,z_min],[x_min,y_max,z_min]])
    vertices_2 = np.array([[x_min,y_min,z_max],[x_max,y_min,z_max],[x_max,y_max,z_max],[x_min,y_max,z_max]])
    vertices_3 = np.array([[x_min,y_min,z_min],[x_min,y_min,z_max],[x_max,y_min,z_max],[x_max,y_min,z_min]])
    vertices_4 = np.array([[x_min,y_max,z_min],[x_min,y_max,z_max],[x_max,y_max,z_max],[x_max,y_max,z_min]])
    vertices_5 = np.array([[x_min,y_min,z_min],[x_min,y_max,z_min],[x_min,y_max,z_max],[x_min,y_min,z_max]])
    vertices_6 = np.array([[x_max,y_min,z_min],[x_max,y_max,z_min],[x_max,y_max,z_max],[x_max,y_min,z_max]])
    list_vertices = [vertices_1,vertices_2,vertices_3,vertices_4,vertices_5,vertices_6]
    faces = Poly3DCollection(list_vertices, linewidths=1, edgecolors=color,label=label)
    faces.set_facecolor(color)

    ax.add_collection3d(faces)

    return ax
 
        

def add_object(objects,dim_of_space,min_max_size_cubes):
    while True:
        x = dim_of_space[0] + np.random.rand()*(dim_of_space[3] - dim_of_space[0])
        y = dim_of_space[1] + np.random.rand()*(dim_of_space[4] - dim_of_space[1])
        z = dim_of_space[2] + np.random.rand()*(dim_of_space[5] - dim_of_space[2])
        x_size = min_max_size_cubes[0] + np.random.rand()*(min_max_size_cubes[3] - min_max_size_cubes[0])
        y_size = min_max_size_cubes[1] + np.random.rand()*(min_max_size_cubes[4] - min_max_size_cubes[1])
        z_size = min_max_size_cubes[2] + np.random.rand()*(min_max_size_cubes[5] - min_max_size_cubes[2])
        cube = np.array([x,y,z,x_size,y_size,z_size])

        if check_cube_not_overlap(cube,objects):
            break
    return cube

def check_cube_not_overlap(cube,objects):
    if not objects:
        return True
    else:
        for key in objects:
            cond_1 = cube[3] < objects[key][0]
            cond_2 = cube[0] > objects[key][3]
            cond_3 = cube[4] < objects[key][1]
            cond_4 = cube[1] > objects[key][4]
            cond_5 = cube[5] < objects[key][2]
            cond_6 = cube[2] > objects[key][0]
            if not cond_1 and not cond_2 and not cond_3 and not cond_4 and not cond_5 and not cond_6:
                return False

        
        else:
            return True

def generate_pose(dim_of_space,objects):
    while True:
        x = dim_of_space[0] + np.random.rand()*(dim_of_space[3] - dim_of_space[0])
        y = dim_of_space[1] + np.random.rand()*(dim_of_space[4] - dim_of_space[1])
        z = dim_of_space[2] + np.random.rand()*(dim_of_space[5] - dim_of_space[2])
        for key in objects:
            if not x > objects[key][3] and not x < objects[key][0] and not y > objects[key][4] and not y < objects[key][1] and not z > objects[key][5] and not z < objects[key][2]:
                break
        return np.array([x,y,z,np.random.rand()])


def write_to_csv(number,current_pose,target_pose,objects,text_instruction,path):
    row = [number,current_pose,target_pose,objects,text_instruction]
    with open('{}/data.csv'.format(path), 'a') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerow(row) 

def main():
    dim_of_space = np.array([0.,0,0.,3,4,2.5])
    min_max_size_cubes = np.array([0.4,0.4,0.4,2.,2.,1.])

    path = os.path.dirname(os.path.realpath(__file__)) + '/training_data'

    for i in range(2):
        objects = {}

        for j in range(3):
            new_object = add_object(objects,dim_of_space,min_max_size_cubes)
            objects[j] = new_object

        target_pose = generate_pose(dim_of_space,objects)
        current_pose = generate_pose(dim_of_space,objects)
        plot(objects,current_pose,target_pose,dim_of_space,path,i)
        text_instruction = input('Describe the target pose: ')
        write_to_csv(i,current_pose,target_pose,objects,text_instruction,path)

main()