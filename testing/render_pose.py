import bpy
import numpy as np
import sys


def set_camera_6dof(scene,x,y,z,rx,ry,rz):
    scene.camera.location.x = x
    scene.camera.location.y = y
    scene.camera.location.z = z
    scene.camera.rotation_euler[0] = rx*(np.pi/180.0)
    scene.camera.rotation_euler[1] = ry*(np.pi/180.0)
    scene.camera.rotation_euler[2] = rz*(np.pi/180.0)


def render_pose(pose,path,render_name):
    #pose should be numpy array with first three numbers x,y,z coords and 4th number rotation between 0 and 1 
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

    rx = 90.
    ry = 0
    pose = pose.round(4)

    x = pose[0]
    y = pose[1]
    z = pose[2]
    rz = 360 * pose[3]
    
    set_camera_6dof(scene,x,y,z,rx,ry,rz)
    scene.render.filepath = path + '/' + render_name
    bpy.ops.render.render( write_still=True )
    return render_name

def main():
    print('tete')
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    print(argv)
    render_pose(np.array([float(argv[0]),float(argv[1]),float(argv[2]),float(argv[3])]),argv[4],argv[5])

main()

