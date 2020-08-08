import os
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
import glob
import pickle
import cv2
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath("/scratches/robot_2/fml35/mphil_project/navigation/opensfm"))

from cropping.equirectangular_crops import equirectangular_crop,euler_to_mat

# with open(os.path.abspath('/data/cvfs/fml35/own_datasets/localisation/dataset_01_room_books/reconstruction.json'), 'r') as fp:
#     original_dict = json.load(fp)[0]

# new_dict = {}
# for image in original_dict["shots"]:
#     new_dict[image] = {}
#     new_dict[image]["rotation"] = original_dict["shots"][image]["rotation"]
#     translation = - R.from_rotvec(original_dict["shots"][image]["rotation"]).inv().apply(np.array(original_dict["shots"][image]["translation"]))
#     new_dict[image]["translation"] = list(translation)

# print(new_dict)
# with open('/data/cvfs/fml35/own_datasets/localisation/dataset_01_room_books/sfm_rotation_translation_in_world_coordinates.json', 'w') as file:
#     json.dump(new_dict, file,indent = 4)

path_to_reconstruction_json = '/data/cvfs/fml35/own_datasets/localisation/new_room/reconstruction.json'
path_to_original_image_folder = '/data/cvfs/fml35/own_datasets/localisation/new_room/images'

path_to_new_image_folder = '/data/cvfs/fml35/own_datasets/localisation/new_room/crops/exp_02_date_08_08_20/cropped_images'
path_to_new_json_file = '/data/cvfs/fml35/own_datasets/localisation/new_room/crops/exp_02_date_08_08_20/data.json'

number_orientations_per_image = 10

with open(path_to_reconstruction_json, 'r') as fp:
    original_dict = json.load(fp)[0]

with open('../../mask_dict.pkl',"rb") as file:
    all_masks = pickle.load(file)

new_dict = {}


# world_rotation = R.from_euler('z',298)
# world_translation = np.array([1.7,3.6,0.64])
# world_scaling = 0.0029

# final_z_degree_allignment = [0,0,-45] 

# params={
#          'batch_size':1,
#          'wrap':True,
#          'H_res': 512,
#          'W_res': 512,
#          'plane_f':0.05,
#          'HFoV':(90 / 360) * 2* np.pi,
#          'VFoV': (90 / 360) *np.pi * 2,
#          'angles': [0,np.pi/2,0] #[x,y,z] # vary just rotation of z, z = 0 corresponnds to center crop, increasing z will shift crop to the right
#         }


for j,image_name in tqdm(enumerate(sorted(original_dict["shots"]))):#['front_frame_01050.png','left_frame_01000.png','right_frame_01000.png','back_frame_01150.png'])
    #img = cv2.imread(path_to_original_image_folder +'/'+ image_name)
    img = Image.open(path_to_original_image_folder +'/'+ image_name)
    # img.save(path_to_new_image_folder + '/' + image_name)
    #img = cv2.imread(path_to_original_image_folder +'/'+ image_name)
    #cv2.imwrite(path_to_new_image_folder + '/' + image_name,img)
    
    sfm_t = original_dict["shots"][image_name]["translation"]
    sfm_r = original_dict["shots"][image_name]["rotation"]

    sfm_rotation = R.from_rotvec(sfm_r)
    new_t = - sfm_rotation.inv().apply(sfm_t)
    orientation_before_crop = list(sfm_rotation.inv().as_euler('xyz',degrees=True))

    if 'front' in image_name:
        center = 182
    elif 'left' in image_name:
        center = 146
    elif 'right' in image_name:
        center = 100
    elif 'back' in image_name:
        center = 78
    
    # x is down/ up, y is tilt, z is rotation around vertical
    if orientation_before_crop[0] > -95 and orientation_before_crop[0] < -85 and orientation_before_crop[1] > - 5 and orientation_before_crop[1] < 5:
        for i in range(-45,45,3):
            new_image_name = image_name.replace('.png',"") + '_crop_' +str(i) + '.png'

            new_r = sfm_rotation.inv()

            new_dict[new_image_name] = {}
            new_dict[new_image_name]["sfm_rotation"] = sfm_r
            new_dict[new_image_name]["sfm_translation"] = sfm_t
            new_dict[new_image_name]["orientation_before_crop"] = orientation_before_crop
            new_dict[new_image_name]["z_after_crop"] = (orientation_before_crop[2] - center - i)%360

            new_dict[new_image_name]["translation"] = list(new_t)



            # params['angles'] = [0,np.pi/2,-(180 - center - i) * np.pi/180]
            # params['R'] = euler_to_mat(z=params['angles'][2],y=params['angles'][1],x=params['angles'][0])
            
            #result = equirectangular_crop(img,params)

            #cv2.imwrite(path_to_new_image_folder + '/' + new_image_name,result)
            # new_dict[new_image_name]["room_orientation"] = list(new_r.as_euler('zyx',degrees=True) + [0,0,i] - final_z_degree_allignment)
            # new_dict[new_image_name]["room_position"] = list(world_scaling * world_rotation.apply(new_t) + world_translation)


            #result = equirectangular_crop(img,params)
            m = all_masks[-(180 - center - i)]
            cropped_img = img.crop((m[1][0], m[0][0], m[1][1], m[0][1]))
            resized_img = cropped_img.resize((128,128))
            # cv2.imwrite(path_to_new_image_folder + '/' + new_image_name,result)
            resized_img.save(path_to_new_image_folder + '/' + new_image_name)



# points = [182,146,100,78]

# points = [182,236,100,78]
# views = ['front','left','right','back']
# params={
#          'batch_size':1,
#          'wrap':True,
#          'H_res': 512,
#          'W_res': 512,
#          'plane_f':0.05,
#          'HFoV':(90 / 360) * 2* np.pi,
#          'VFoV': (90 / 360) *np.pi * 2,
#          'angles': [0,np.pi/2,0] #[x,y,z] # vary just rotation of z, z = 0 corresponnds to center crop, increasing z will shift crop to the right
#         }

# for j,image_name in enumerate(['front_frame_01000.png','left_frame_01000.png','right_frame_01000.png','back_frame_01000.png']):
#     params['angles'] = [0,np.pi/2,-(180 - points[j]) * np.pi/180]
#     params['R'] = euler_to_mat(z=params['angles'][2],y=params['angles'][1],x=params['angles'][0])
    
#     img = cv2.imread(path_to_original_image_folder +'/'+ image_name)
#     cv2.imwrite(path_to_new_image_folder + '/' + image_name,img)
#     result = equirectangular_crop(img,params)

#     new_image_name = image_name.replace('.png',"") + views[j] + '.png'
#     cv2.imwrite(path_to_new_image_folder + '/' + new_image_name,result)

# r1 = R.from_euler('xyz', [180,180,90],degrees=True)
# r2 = R.from_euler('z',90)
# print((r2 * r1).as_euler('zyx',degrees=True))

# I dont understand how to combine with orientation.
# When I looked at just osfm rotation.inv().as_euler('xyz',degrees="True")' seemed to make sense
# Go back to this and just subtract angles from z


with open(path_to_new_json_file, 'w', encoding='utf-8') as f:
    json.dump(new_dict, f, ensure_ascii=False, indent=4)