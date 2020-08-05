# ok so have pixel coords now use width, height and focal length and depthmaps to project transform into
#camera coords

# focal length

import numpy as np
from scipy.spatial.transform import Rotation as scipy_rotation
import json


# load depth maps and colors
npzfile = np.load('masked_world_points_depths_colors.npz')
masked_colors,masked_depths,original_world_coordinates = npzfile["masked_colors"],npzfile["masked_depths"],npzfile["world_coordinates"]

plane_f = 0.05
HFoV = 0.8
VFoV = 0.8

plane_w = (2*plane_f)*np.tan(HFoV/2.0)
plane_h = (2*plane_f)*np.tan(VFoV/2.0)

#transform into camera coordinates
#TODO: not sure which way round
pixel_w = masked_colors.shape[1]
pixel_h = masked_colors.shape[0]

coords_on_image_plane = np.zeros((masked_colors.shape[0],masked_colors.shape[1],3))

for i in range(masked_colors.shape[0]):
    for j in range(masked_colors.shape[1]):
        coords_on_image_plane[i,j] = np.array([(i-pixel_w/2)*plane_w/2,(j-pixel_h/2)*plane_h/2,plane_f])

# Now scale coordinates so that they lie on unit sphere
sphere_coordinates = coords_on_image_plane / np.expand_dims(np.linalg.norm(coords_on_image_plane,axis=2),axis=2)

# now multiply coordinates by their depth values to get in camera coordinates (but still need to perform inverse of cropping rotation)
camera_coordinates = sphere_coordinates * np.expand_dims(masked_depths,axis=2)

# undo cropping rotation

# apply camera rotation and translation
with open('/data/cvfs/fml35/own_datasets/localisation/dataset_01_room_books/reconstruction.json', 'r') as file:
    reconstruction_dict = json.load(file)[0]


camera_sfm = reconstruction_dict["shots"]["facing_bookshelf_part2_frame_00100.png"]

#Find rotation, R = inv(R_osfm)
sfm_rotation = scipy_rotation.from_rotvec(np.array(camera_sfm["rotation"]))
R = sfm_rotation.inv().as_matrix()

# find translation T = -inv(R_osfm) * T_osfm
T = -sfm_rotation.inv().apply(np.array(camera_sfm["translation"]))

world_coordinates = np.matmul(R,np.expand_dims(camera_coordinates,axis=3)) + T


def write_as_ply(file_path,world_coordinates,colors):
    assert (world_coordinates.shape[0] == colors.shape[0], 'shapes dont match')
    with open(file_path, 'w') as file:
        file.write('ply\nformat ascii 1.0\n\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nend_header\n'.format(colors.shape[0]))
        for i in range(world_coordinates.shape[0]):
            file.write('{} {} {} {} {} {}\n'.format(world_coordinates[i,0],world_coordinates[i,1],world_coordinates[i,2],colors[i,0],colors[i,1],colors[i,2]))


write_as_ply('original_world_coordinates.ply',original_world_coordinates,colors)
write_as_ply('transformed_world_coordinates.ply',original_world_coordinates,colors)

print(coords_on_image_plane[100,:3])
print(sphere_coordinates[100,:3])
print(masked_depths[100,:])
print(camera_coordinates[100,:30])
print(world_coordinates)