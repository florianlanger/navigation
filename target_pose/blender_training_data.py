import numpy as np
import os
import csv
import socket
import pickle
import sounddevice as sd
import json

       

def add_object(objects,dim_of_space,min_max_size_cubes):
    while True:
        x_size = min_max_size_cubes[0] + np.random.rand()*(min_max_size_cubes[3] - min_max_size_cubes[0])
        y_size = min_max_size_cubes[1] + np.random.rand()*(min_max_size_cubes[4] - min_max_size_cubes[1])
        z_size = min_max_size_cubes[2] + np.random.rand()*(min_max_size_cubes[5] - min_max_size_cubes[2])
        z = z_size/2
        x = dim_of_space[0] + x_size/2 + np.random.rand()*(dim_of_space[3] - dim_of_space[0] - x_size)
        y = dim_of_space[1] + y_size/2 + np.random.rand()*(dim_of_space[4] - dim_of_space[1] - y_size)
        cube = np.array([x,y,z,x_size,y_size,z_size])

        if check_cube_not_overlap(cube,objects):
            break
    return cube

def check_cube_not_overlap(cube,objects):
    if not objects:
        return True
    else:
        for key in objects:
            cond_1 = cube[0] + 0.5 *cube[3] < objects[key][0] - 0.5*objects[key][3]
            cond_2 = cube[0] - 0.5 *cube[3] > objects[key][0] + 0.5*objects[key][3]
            cond_3 = cube[1] + 0.5 *cube[4] < objects[key][1] - 0.5*objects[key][4]
            cond_4 = cube[1] - 0.5 *cube[4] > objects[key][1] + 0.5*objects[key][4]
            cond_5 = cube[2] + 0.5 *cube[5] < objects[key][2] - 0.5*objects[key][5]
            cond_6 = cube[2] - 0.5 *cube[5] > objects[key][2] + 0.5*objects[key][5]
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
            if not x > objects[key][0] + 0.5 * objects[key][3] and not x < objects[key][0] - 0.5 * objects[key][3] and not y > objects[key][1] + 0.5 * objects[key][4] and not y < objects[key][1] - 0.5 * objects[key][4] and not z > objects[key][2] + 0.5 * objects[key][5] and not z < objects[key][2] - 0.5 * objects[key][5]:
                break
        return np.array([x,y,z,np.random.rand()])

def orient_current_pose(current_pose,target_pose):

    if (target_pose[0]-current_pose[0]) > 0 and (target_pose[1]-current_pose[1]) > 0:
        angle = - np.arctan((target_pose[0]-current_pose[0])/(target_pose[1]-current_pose[1]))
    elif (target_pose[0]-current_pose[0]) > 0 and (target_pose[1]-current_pose[1]) < 0:
        angle = -np.pi / 2  - np.arctan(-(target_pose[1]-current_pose[1])/(target_pose[0]-current_pose[0]))
    elif (target_pose[0]-current_pose[0]) < 0 and (target_pose[1]-current_pose[1]) > 0:
        angle = np.arctan(-(target_pose[0]-current_pose[0])/(target_pose[1]-current_pose[1]))
    elif (target_pose[0]-current_pose[0]) < 0 and (target_pose[1]-current_pose[1]) < 0:
        angle = np.pi / 2 + np.arctan((target_pose[1]-current_pose[1])/(target_pose[0]-current_pose[0]))


    current_pose[3] = angle/ (2*np.pi)
    return current_pose


def write_to_csv(training_dict,path):
    with open(path, 'a') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerow(str(training_dict)) 

def main():


    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65434     # The port used by the server

    dim_of_space = np.array([0.,0,0.,3,4,2.5])
    min_max_size_cubes = np.array([0.4,0.4,0.4,2.,2.,1.])

    path = os.path.dirname(os.path.realpath(__file__)) + '/training_data'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT)) 
        print('Connected to server')

        while True:
            objects = {}

            with open(path + '/counter.txt', "r") as file:
                counter = file.readline()

            for j,color in enumerate(['blue','red','green']):
                new_object = add_object(objects,dim_of_space,min_max_size_cubes)
                objects[color] = list(new_object)

            target_pose = generate_pose(dim_of_space,objects)
            current_pose = generate_pose(dim_of_space,objects)
            current_pose = orient_current_pose(current_pose,target_pose)
            training_dict = {'ID':str(counter).zfill(4),'objects':objects,'current_pose':list(current_pose),'target_pose':list(target_pose),'dim_of_space':list(dim_of_space),'path':path}
            print(training_dict)
            s.sendall(pickle.dumps(training_dict))
            while True:
                data = s.recv(1024)
                if data == b'done':
                    break
            input('Type anything to start recording: ')
            print('Start recording ...')
            duration = 5  # seconds
            fs = 44100
            myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            print('Finished recording')
            np.save(path+'/recordings/recording_{}.npy'.format(str(counter).zfill(4)),myrecording)
            json.dump( training_dict, open('{}/data.csv'.format(path), 'a' ))

            with open(path + '/counter.txt', "w") as file:
                file.write(str(int(counter)+1))
main()