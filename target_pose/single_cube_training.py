import numpy as np
import os
import csv
import socket
import pickle
import sounddevice as sd
import json
# import speech_recognition as sr
from scipy.io.wavfile import write
import wavio


import azure.cognitiveservices.speech as speechsdk

speech_key, service_region = '4203927a90be4b1785cee6bdd8310f48', "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)


result = speech_recognizer.recognize_once()


def add_cube():
    size = 1.
    x = 2. + np.random.rand()*4
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

def transcribe(counter,path):

    r = sr.Recognizer()
    myarray = np.load(path + '/recordings/recording_{}.npy'.format(str(counter).zfill(4)))
    wavio.write(path + '/recordings/recording_{}.wav'.format(str(counter).zfill(4)), myarray, 44100 ,sampwidth=1)
    harvard = sr.AudioFile(path + '/recordings/recording_{}.wav'.format(str(counter).zfill(4)))
    with harvard as source:
        audio = r.record(source)
    transcription = r.recognize_google(audio,language='en-IN',show_all = False)
    print(transcription)
    return transcription

def transcribe_azure(speech_config,counter,path):

    myarray = np.load(path + '/recordings/recording_{}.npy'.format(str(counter).zfill(4)))
    wavio.write(path + '/recordings/recording_{}.wav'.format(str(counter).zfill(4)), myarray, 44100 ,sampwidth=1)

    audio_input = speechsdk.audio.AudioConfig(filename=path + '/recordings/recording_{}.wav'.format(str(counter).zfill(4)))

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    result = speech_recognizer.recognize_once()
    print(result.text)
    return result.text

def main():
    
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65440     # The port used by the server


    path = os.path.dirname(os.path.realpath(__file__)) + '/training_data1'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT)) 
        print('Connected to server')

        #while True:
        for i in range(100):

            with open(path + '/counter.txt', "r") as file:
                counter = file.readline()
            cube = add_cube()

            target_pose = generate_pose(cube)

            training_dict = {'ID':str(counter).zfill(4),'cube':list(cube),'target_pose':list(target_pose)}
            s.sendall(pickle.dumps(training_dict))
            while True:
                data = s.recv(1024)
                if data == b'done':
                    break
            #description = input('Describe the pose: ')
            input('Press anything to start recording')
            print('Start recording ...')
            duration = 5  # seconds
            fs = 44100
            myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            print('Finished recording')
            np.save(path+'/recordings/recording_{}.npy'.format(str(counter).zfill(4)),myrecording)

            #transcription = transcribe_azure(speech_config,counter,path)


            with open('{}/data.csv'.format(path), 'a' ) as file:
                if counter == "0":
                    file.write('{},{},{},{},{}\n'.format('ID','cube','target_pose','transcription','correction'))
                file.write('{},{},{},\n'.format(training_dict["ID"],training_dict["cube"],training_dict["target_pose"]))
            with open(path + '/counter.txt', "w") as file:
                file.write(str(int(counter)+1))
main()