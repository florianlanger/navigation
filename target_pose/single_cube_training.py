import numpy as np
import os
import csv
import socket
import pickle
import sounddevice as sd
import json
import speech_recognition as sr
from scipy.io.wavfile import write
import wavio   

def add_cube(min_max_size_cubes):
    size = min_max_size_cubes[0] + np.random.rand()*(min_max_size_cubes[1] - min_max_size_cubes[0])

    x = 1 + np.random.rand()*4
    y = - x +  np.random.rand()* 2 * x
    z = np.random.rand()*x
    cube = np.array([x,y,z,size,size,size])
    return cube


# Have 9 x 9 x 9 array where first number is possible points in the x direction, second in y, third in z
#point 5 x 5 x 5 is center of the cube
# one corner is 4 x 4 x 4 
def generate_pose(cube):
    while True:
        indices = np.random.randint(9,size=(3))
        position = indices_to_position(cube,indices)
        if np.abs(position[1]) < position[0] and position[2] < position[0] and position[2] > 0:
            return position

def indices_to_position(cube,indices):
    position = cube[:3] + (indices - 4) * cube[3:6]/2
    return position


def write_to_csv(training_dict,path):
    with open(path, 'a') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerow(str(training_dict)) 

def main():


    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65440     # The port used by the server

    min_max_size_cubes = np.array([0.4,2.])

    path = os.path.dirname(os.path.realpath(__file__)) + '/training_data'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT)) 
        print('Connected to server')

        #while True:
        for i in range(50):

            with open(path + '/counter.txt', "r") as file:
                counter = file.readline()

            cube = add_cube(min_max_size_cubes)

            target_pose = generate_pose(cube)

            training_dict = {'ID':str(counter).zfill(4),'cube':list(cube),'target_pose':list(target_pose)}
            s.sendall(pickle.dumps(training_dict))
            while True:
                data = s.recv(1024)
                if data == b'done':
                    break
            description = input('Describe the pose: ')
            # print('Start recording ...')
            # duration = 5  # seconds
            # fs = 44100
            # myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            # sd.wait()
            # print('Finished recording')
            # np.save(path+'/recordings/recording_{}.npy'.format(str(counter).zfill(4)),myrecording)
            training_dict['description'] = description
            json.dump(training_dict, open('{}/data.csv'.format(path), 'a' ))

            with open('{}/data.csv'.format(path), 'a' ) as file:
                file.write('\n')
            with open(path + '/counter.txt', "w") as file:
                file.write(str(int(counter)+1))
main()