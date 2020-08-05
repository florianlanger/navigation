import os
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
import glob

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


os.chdir("/data/cvfs/fml35/own_datasets/localisation/dataset_01_room_books/center_cropped_images")

list_images = glob.glob('*.png')
os.chdir("/data/cvfs/fml35/own_datasets/localisation/dataset_01_room_books/center_cropped_images_small")

for image_name in list_images:
    image = Image.open('../center_cropped_images/' + image_name, mode = 'r')
    #image = image.crop((1519,629,1519+882,629 + 662))
    image = image.resize((200,150))
    image.save(image_name)