from os import listdir
from PIL import Image

occluder_names = listdir("/data/cvfs/fml35/original_downloads/coco/train2017/")

basewidth = 100
counter = 0
for i in range(93606):
    try:
        img = Image.open('/data/cvfs/fml35/original_downloads/coco/train2017/' + occluder_names[i])
        aspect_ratio = float(img.size[0])/img.size[1]
        if aspect_ratio >= 1.:
            new_height = 100
            new_width = int(aspect_ratio * 100)
        else:
            new_width = 100
            new_height = int(100/aspect_ratio)

        img = img.resize((new_width,new_height), Image.ANTIALIAS)
        img.save('/data/cvfs/fml35/derivative_datasets/coco/train2017/images/{}.jpg'.format(str(counter).zfill(6)))
        counter += 1
    except:
        print("Corrupted image")