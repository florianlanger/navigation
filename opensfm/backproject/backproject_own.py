import numpy as np
import glob
from plyfile import PlyData, PlyElement
import sys
import os
import cv2
from PIL import Image
import json
import pickle
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as scipy_rotation
from tqdm import tqdm

from camera_types import denormalized_image_coordinates,get_shot_camera_data,load_equirectangular_camera,normalized_image_coordinates


def filter_closest(source_world_points,source_camera_points,source_sphere_points,source_points_ids,source_colors,source_mesh_ids,width,height):

    # idx works on np.array and not lists.
    camera_dist =np.reshape(np.linalg.norm(source_camera_points,axis=-1),(-1)) 
    dist_sorted_index = np.argsort(camera_dist)
    mesh_ids = source_mesh_ids[dist_sorted_index,:]
    pts_ids = source_points_ids[dist_sorted_index]
    colors = source_colors[dist_sorted_index,:]
    world_pts = source_world_points[dist_sorted_index,:]
    pts_camera = source_camera_points[dist_sorted_index,:]
    pts_camera_sphere = source_sphere_points[dist_sorted_index,:]


    pts_ids_unique_info=np.unique(pts_ids, return_index=True)
    pts_ids_unique=pts_ids_unique_info[0]
    pts_ids_unique_index=pts_ids_unique_info[1]


    image_colors = np.zeros(shape=[height*width, 3],dtype=np.uint8)
    points_3d_camera = np.zeros(shape=[height*width, 3],dtype=np.float32)
    points_3d_sphere =np.zeros(shape=[height*width, 3],dtype=np.float32)
    points_3d_world = np.zeros(shape=[height*width,3],dtype=np.float32)

    result_mesh_ids = np.zeros(shape=[height*width,1],dtype=np.int32)
    mask = np.zeros(shape=[height*width,1],dtype=np.uint8)

    image_colors[pts_ids_unique, :] = colors[pts_ids_unique_index,:]
    mask[pts_ids_unique,:]=255
    points_3d_camera[pts_ids_unique,:]=pts_camera[pts_ids_unique_index,:]
    points_3d_sphere[pts_ids_unique,:]= pts_camera_sphere[pts_ids_unique_index,:]
    points_3d_world[pts_ids_unique,:] =world_pts[pts_ids_unique_index,:]
    result_mesh_ids[pts_ids_unique,:]=mesh_ids[pts_ids_unique_index,:]

    shp =(height,width,3)
    image_colors=np.reshape(image_colors,shp)
    mask =np.reshape(mask,(height,width,1))
    points_3d_camera =np.reshape(points_3d_camera,shp)
    points_3d_sphere= np.reshape(points_3d_sphere,shp)
    points_3d_world=np.reshape(points_3d_world,shp)

    result_mesh_ids = np.reshape(result_mesh_ids,(height,width,1))
    return points_3d_world,points_3d_camera,points_3d_sphere,image_colors,mask,result_mesh_ids
    # np.savez_compressed('../points_3d_info_new/'+short_file_name.split('.')[0]+'.npz',image_colors=image_colors,mask=mask,points_3d_camera=points_3d_camera,points_3d_sphere_unmasked=opensfm_sphere_coords,points_3d_sphere=points_3d_sphere,points_3d_world=points_3d_world,R=RT[:3,:3],T=RT[:3,3])
    #exit(1)



x_width=3840
x_height=1920
pixel_coords= np.zeros((x_height,x_width,2))

print('Filling coordinates')
for row in range(x_height):
	for col in range(x_width):
		pixel_coords [row,col,1]=row +0.5
		pixel_coords[row,col,0]=col +0.5
norm_pixel_coords= normalized_image_coordinates(np.reshape(pixel_coords,(x_width*x_height,2)),1920,960)

with open('../cropping/reconstruction.json', 'r') as file:
    reconstruction_dict = json.load(file)[0]

norm_pixel_coords= normalized_image_coordinates(np.reshape(pixel_coords,(x_width*x_height,2)),1920,960)


# only use 10 points for debugging
n = 19000000
#n = len(reconstruction_dict["points"])
# load mesh
mesh_data = {}
mesh_data['points'] = np.zeros((n,3))
mesh_data['colors'] = np.zeros((n,3))
mesh_data['mesh_ids'] = np.reshape( np.arange(n,dtype=np.int32),(n,1))

counter = 0
with open('../merged.ply', 'r') as file:
    for i in tqdm(range(n + 14 + 1)):
        line = file.readline().split()
        #print(line)
        if i > 14:
            mesh_data['points'][counter] = [line[0],line[1],line[2]]
            mesh_data['colors'][counter] = [line[6],line[7],line[8]]
            counter += 1

#print(mesh_data)

with open('../../pose_room_books/mask_dict.pkl',"rb") as file:
    all_masks = pickle.load(file)

# step R and t from camera 
for i,image_name in enumerate(['facing_bookshelf_part2_frame_00950.png', 'facing_bookshelf_part2_frame_00520.png']):
    

	# not sure here if t straight from file or need to rotate
    rotation = np.array(reconstruction_dict["shots"][image_name]["rotation"])
    translation = np.array(reconstruction_dict["shots"][image_name]["translation"])

    # Think that have - here from how opensfm define rotation, but apparently not
    abstract_rotation = scipy_rotation.from_rotvec(rotation)

    for i in range(-150,150,50):
        new_image_name = image_name.replace('.png',"") + '_crop_z_' + str(i) + '.png'
        crop_rotation = scipy_rotation.from_euler('xyz',[0,i,0],degrees=True)
        R = abstract_rotation.as_matrix()
        
        # not sure here if t straight from file or need to rotate
        t = translation

        shot_camera = load_equirectangular_camera({'width':x_width,'height':x_height})

        result_rgb = np.zeros((x_height,x_width,3))
        result_label = np.zeros((x_height,x_width,3))

        source_world_points = mesh_data['points']
        source_image_colors = mesh_data['colors']
        source_meshids = mesh_data['mesh_ids']

        mul = np.matmul(R,np.reshape(source_world_points,(source_world_points.shape[0],3,1)))

        source_camera_points= mul[:,:,0]+np.reshape(t,(1,3))

        source_camera_points = crop_rotation.inv().apply(source_camera_points)

        dist_cameras = np.linalg.norm(source_camera_points,axis=-1)
        source_sphere_points =source_camera_points/np.reshape(dist_cameras,(source_camera_points.shape[0],1))
        cam_coords = shot_camera.project_many(source_camera_points)

        denom_cam_coords= denormalized_image_coordinates(cam_coords,shot_camera.width,shot_camera.height)
        #print('denom '+str(denom_cam_coords.shape))
        row = np.int32(denom_cam_coords[:,1])
        col = np.int32(denom_cam_coords[:,0])
        cid = np.arange(col.shape[0],dtype=np.int32)

        source_points_ids  = row * shot_camera.width+col

        points_3d_world,points_3d_camera,points_3d_sphere,image_colors,mask,mesh_ids= filter_closest(source_world_points,source_camera_points, source_sphere_points, source_points_ids,source_image_colors,source_meshids,shot_camera.width,shot_camera.height)

        # npzfile = np.load('../cropping/mask.npz')
        # mask_indices = npzfile["mask_as_start_and_end_indices"]
        
        # masked_depths = np.linalg.norm(points_3d_camera.copy(),axis=2)[mask_indices[0,0]:mask_indices[0,1],mask_indices[1,0]:mask_indices[1,1]]
        # masked_colors = image_colors[mask_indices[0,0]:mask_indices[0,1],mask_indices[1,0]:mask_indices[1,1],:]
        # masked_world_points = points_3d_world[mask_indices[0,0]:mask_indices[0,1],mask_indices[1,0]:mask_indices[1,1],:]
        
        # np.savez('masked_world_points_depths_colors.npz',masked_world_points=masked_world_points,masked_depths=masked_depths,masked_colors=masked_colors)

        sphere_coords = np.reshape(shot_camera.pixel_bearing_many(norm_pixel_coords),(shot_camera.height,shot_camera.width,3))
        assert (np.min(np.abs(np.linalg.norm(sphere_coords,axis=-1)-1))<0.001)
        sphere_dist = np.sqrt(np.sum(np.square( sphere_coords-points_3d_sphere	),axis=-1))
        sphere_dist [mask[:,:,0]==0]=0
        #plt.imshow(sphere_dist)
        #plt.show()
        #assert (np.max(sphere_dist)<0.004), 'Max dist '+str(np.max(sphere_dist))

        print('distance check is finished')
        result_rgb[row,col,:]=source_image_colors[cid,::-1]

        m = all_masks[0]
        # image  = Image.fromarray(result_rgb)
        result_rgb = result_rgb[m[0][0]:m[0][1],m[1][0]:m[1][1],:]
        # cropped_img = image.crop((m[1][0], m[0][0], m[1][1], m[0][1])) 
        # cropped_img.save('../../pose_room_books/images_backprojected/' + new_image_name)

        cv2.imwrite('../../pose_room_books/images_backprojected/' + new_image_name ,result_rgb)

    
    
print('Finished')


