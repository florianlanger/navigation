import random as ra
from scipy.spatial.transform import Rotation as R
import os
import bpy
import csv
import numpy as np


def set_camera_6dof(x,y,z,rx,ry,rz):
    scene.camera.location.x = x
    scene.camera.location.y = y
    scene.camera.location.z = z
    scene.camera.rotation_euler[0] = rx*(pi/180.0)
    scene.camera.rotation_euler[1] = ry*(pi/180.0)
    scene.camera.rotation_euler[2] = rz*(pi/180.0)

def write_render_to_csv(path,render_name,x,y,z,rz):
    row = [render_name,x,y,z,rz]
    with open('{}/positions.csv'.format(path), 'a') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerow(row)

def main(path):
    rx = 90.
    ry = 0
    for number,kind in zip([30,5],['train','val']):
        dir_path = path + '/' + kind
        for counter in range(number):
            x = np.round(np.random.uniform(-1.30,2.30),3)
            y = np.round(np.random.uniform(-0.5,1.5),3)
            z = np.round(np.random.uniform(0.2,1.8),3)
            rz = np.round(np.random.uniform(0.0,360.),3)
            render_name = 'render_{}_x_{}_y_{}_z_{}_rz_{}.png'.format(counter,x,y,z,rz)
            write_render_to_csv(dir_path,render_name,x,y,z,rz)
            set_camera_6dof(x,y,z,rx,ry,rz)
            scene.render.filepath = dir_path + '/images/' + render_name
            bpy.ops.render.render( write_still=True )


if __name__ == "__main__":


    full_path = '/data/cvfs/fml35/own_datasets/grid_world/ignas_living_room_random_poses'
    os.mkdir(full_path)
    os.mkdir(full_path + '/train')
    os.mkdir(full_path + '/val')
    os.mkdir(full_path + '/train/images')
    os.mkdir(full_path + '/val/images')

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.max_bounces = 12
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 1
    bpy.context.scene.cycles.transmission_bounces = 1
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.render.tile_x = 10
    bpy.context.scene.render.tile_y = 10
    bpy.context.scene.cycles.samples = 128




    #set global parameter for blender
    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_x = 100
    scene.render.resolution_y = 100
    fov = 50.0
    pi = 3.14159265
    # Set camera fov in degrees
    scene.camera.data.angle = fov*(pi/180.0)
    # Set camera rotation in euler angles
    scene.camera.rotation_mode = 'XYZ'
    
   
    
    
    #main(full_path)