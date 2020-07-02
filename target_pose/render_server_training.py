import bpy
import numpy as np
import sys
import socket
import pickle


def set_camera_6dof(scene,current_pose):
    scene.camera.location.x = current_pose[0]
    scene.camera.location.y = current_pose[1]
    scene.camera.location.z = current_pose[2]
    scene.camera.rotation_euler[0] = 90.
    scene.camera.rotation_euler[1] = 0.
    scene.camera.rotation_euler[2] = current_pose[3]*360



def main():

    # define scene parameters
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


    # define server
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65434     # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(4096)
                if repr(data) != "b''":
#                    objects = {"blue": np.array([1.38693551, 2.87597055, 2.23325024, 1.35041096, 0.40994586,
#                       0.89055494]), "green": np.array([1.69331235, 0.38310432, 1.53923688, 0.87343599, 1.74731842,
#                       0.67999495]), "red": np.array([0.68697315, 3.56860321, 0.30755341, 1.11452907, 0.46323963,
#                       0.73184977])}
#                    training_dict =  {'objects':objects,'current_pose':np.array([1.,2,0.5,0.0]),'target_pose':np.array([0.,-1,0.5,0.5])}
                    
                    #clear previous cubes
                    bpy.ops.object.select_by_type(type='MESH')
                    bpy.ops.object.delete(use_global=False)

                    #add ground plane
                    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, location=(0, 0, 0))
                    bpy.ops.transform.resize(value=(3,3,3))

                    training_dict = pickle.loads(data) 
                    print(training_dict['objects'],training_dict['current_pose'],training_dict['target_pose'])

                    # Adjust camera
                    set_camera_6dof(scene,training_dict['current_pose'])

                    #Add obstacles
                    add_cubes(scene,training_dict['objects'])


                    add_target_pose(scene,training_dict['target_pose'])
                    bpy.context.scene.update()

    # Render the current view
#    scene.render.filepath = path + '/' + render_name
#    bpy.ops.render.render( write_still=True )

                    conn.sendall(b'done')
                elif len(data) == 0: 
                    break


def add_cubes(scene,objects):
    
    color_dict = {'blue': (0.,0.,1.,1.),"green":(0.,1.,0.,1.),"red":(1.,0.,0.,1.)}
    for key in objects:
        x,y,z = objects[key][0],objects[key][1],objects[key][2]
        x_scale,y_scale,z_scale = objects[key][3],objects[key][4],objects[key][5]
        #blender command add object and label with key
        #name by color
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(x,y,z))
        bpy.ops.transform.resize(value=(x_scale,y_scale,z_scale))
        obj = bpy.context.active_object
        material = bpy.data.materials.new(name = 'material')
        obj.data.materials.append(material)
        bpy.context.object.active_material.diffuse_color = color_dict[key]

def add_target_pose(scene,target_pose):
    x,y,z,angle = target_pose[0],target_pose[1],target_pose[2],target_pose[3]
    bpy.ops.mesh.primitive_cone_add(radius1=0.2, radius2=0, depth=0.3, enter_editmode=False, location=(x,y,z),rotation=(90,0.,360*angle))


main()
