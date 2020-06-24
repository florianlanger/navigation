import bpy
import random as ra
from scipy.spatial.transform import Rotation as R
import os
import csv
import numpy as np


def set_camera_6dof(scene,x,y,z,rx,ry,rz):
    scene.camera.location.x = x
    scene.camera.location.y = y
    scene.camera.location.z = z
    scene.camera.rotation_euler[0] = rx*(np.pi/180.0)
    scene.camera.rotation_euler[1] = ry*(np.pi/180.0)
    scene.camera.rotation_euler[2] = rz*(np.pi/180.0)


def render(history_dict,exp_path,scene):
    rx = 90.
    ry = 0
    print(history_dict["counter"])
    for counter in range(history_dict["counter"]):
        x = history_dict["predicted_poses"][counter][0]
        y = history_dict["predicted_poses"][counter][1]
        z = history_dict["predicted_poses"][counter][2]
        rz = history_dict["predicted_poses"][counter][3] * 360.
        set_camera_6dof(scene,x,y,z,rx,ry,rz)
        scene.render.filepath = exp_path + '/renders/' + history_dict["image_names"][counter]
        bpy.ops.render.render( write_still=True )



def main():

    ########### name of experiment ############
    name = 'time_17_32_00_date_24_06_2020'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    exp_path = dir_path + '/../experiments/' + name
    #os.mkdir(exp_path + '/renders')

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.max_bounces = 3
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
    # Set camera fov in degrees
    scene.camera.data.angle = fov*(np.pi/180.0)
    # Set camera rotation in euler angles
    scene.camera.rotation_mode = 'XYZ'
    history_dict = np.load(exp_path + '/history_dict.npy',allow_pickle=True).item()
    
    render(history_dict,exp_path,scene)


    
   
    
    
if __name__ == "__main__":
    main()