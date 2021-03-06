import numpy as np
import os
from datetime import datetime
import cv2
import sys
import json
import time
import shutil


from tellopy.modified_tellopy import Tello

def make_directories(exp_path):
    os.mkdir(exp_path)
    os.mkdir(exp_path + '/images')
    shutil.copytree(exp_path +'/../../../code',exp_path +'/code')


def execute_command(tello,tuple_command):
    if tuple_command[0] == "forward":
        tello.move_forward(tuple_command[1])
    elif tuple_command[0] == "backward":
        tello.move_back(tuple_command[1])
    elif tuple_command[0] == "left":
        tello.move_left(tuple_command[1])
    elif tuple_command[0] == "right":
        tello.move_right(tuple_command[1])
    elif tuple_command[0] == "up":
        tello.move_up(tuple_command[1])
    elif tuple_command[0] == "down":
        tello.move_down(tuple_command[1])
    elif tuple_command[0] == "cw":
        tello.rotate_clockwise(tuple_command[1])
    elif tuple_command[0] == "ccw":
        tello.rotate_counter_clockwise(tuple_command[1])


def save_image(image_name,tello):
    img = tello.get_frame_read().frame
    cv2.imwrite(image_name,img)

def main():

    name = 'callback_right'
    # Bookkeeping 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_path = '{}/../experiments/fly_sequences/{}_{}'.format(dir_path, name,datetime.now().strftime("time_%H_%M_%S_date_%d_%m_%Y"))
    make_directories(exp_path)

    list_of_commands = [['up',20],['up',20],['up',20],['up',20],
    ['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],
    ['cw',90],
    ['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],
    ['cw',90],
    ['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],['forward',20],
    ]

    delay = 2000


    image_names = []
    tello = Tello()
    tello.connect()
    tello.streamon()

    #tello.takeoff()

    start_time = int(round(time.time() * 1000))
    last_action_time = int(round(time.time() * 1000))
    action_counter = 0
    while action_counter < len(list_of_commands):
        current_time = int(round(time.time() * 1000))
        
        if current_time - last_action_time > delay:
            #execute_command(tello,list_of_commands[action_counter])
            action_counter += 1
            last_action_time = int(round(time.time() * 1000))
            print(action_counter)

            image_name = 'image_{}.png'.format(str(action_counter).zfill(6))
            save_image(exp_path + '/images/'+image_name,tello)
        # image_names.append(image_name)
        # with open(exp_path + '/image_names.txt','w') as text_file:
        #     text_file.write(str(image_names))
    

    # tello.land()
    # tello.end()



if __name__ == "__main__":
    main()