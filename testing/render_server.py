import bpy
import numpy as np
import sys
import socket
import pickle


def set_camera_6dof(scene,x,y,z,rx,ry,rz):
    scene.camera.location.x = x
    scene.camera.location.y = y
    scene.camera.location.z = z
    scene.camera.rotation_euler[0] = rx*(np.pi/180.0)
    scene.camera.rotation_euler[1] = ry*(np.pi/180.0)
    scene.camera.rotation_euler[2] = rz*(np.pi/180.0)


def render_pose(pose,path,render_name):
    #pose should be numpy array with first three numbers x,y,z coords and 4th number rotation between 0 and 1

    #set global parameter for blender
    scene = bpy.data.scenes["Scene"]

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

    # define scene parameters
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 1


    print(bpy.context.scene.render.engine)

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

    # define server
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65437      # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if repr(data) != "b''":
                    render_dict = pickle.loads(data)
                    print(render_dict['pose'],render_dict['path'],render_dict['render_name'])
                    render_pose(render_dict['pose'],render_dict['path'],render_dict['render_name'])
                    conn.sendall(b'done')
                elif len(data) == 0: 
                    break

main()

