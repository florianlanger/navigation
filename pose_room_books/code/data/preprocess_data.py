import os
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
import glob
import pickle

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

path_to_reconstruction_json = '/Users/legend98/Google Drive/MPhil project/navigation/opensfm/cropping/reconstruction.json'
path_to_original_image_folder = '/Users/legend98/Desktop'

path_to_new_image_folder = '../../images'
path_to_new_json_file = '../../data.json'

number_orientations_per_image = 10

# with open(path_to_reconstruction_json), 'r') as fp:
#     original_dict = json.load(fp)[0]
with open('../../mask_dict.pkl',"rb") as file:
    all_masks = pickle.load(file)

for image_name in ['location_as_vfov.JPG']:#["facing_bookshelf_part2_frame_00100.png"]:#original_dict["shots"]:
    image = Image.open(path_to_original_image_folder +'/'+ image_name, mode = 'r')
    image = image.resize((3840,1920))
    for i in range(-50,50,5):
        m = all_masks[i]
        cropped_img = image.crop((m[1][0], m[0][0], m[1][1], m[0][1])) 
        cropped_img.save('test_crops/angle_{}.png'.format(i))