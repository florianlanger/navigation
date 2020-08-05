import numpy as np
import json

objects = {
     "drawer": {"dimensions":[120,58,118,350,0,0,],"scaling":[1,1,1]},
    "couch": {"dimensions":[90,204,100,168,120,0],"scaling":[1,1,1]},
    "bed": {"dimensions":[200,186,62,258,140,0],"scaling":[1,1,1]},
    "desk": {"dimensions":[110,50,78,308,418,0],"scaling":[1,1,1]},
    "lamp": {"dimensions":[20,20,80,444,354,46],"scaling":[1,1,1]},
    "bedside table": {"dimensions":[40,40,46,432,340,0],"scaling":[1,1,1]},
    "computer screen": {"dimensions":[64,24,62,330,394,78],"scaling":[1,1,1]},
    "box": {"dimensions":[54,96,38,0,190,0],"scaling":[1,1,1]}
}

new_dict = {}
for object in objects:
    new_dict[object] = {}
    dims = np.zeros(6)
    dims[0:3] = objects[object]["dimensions"][3:6] + 0.5 * np.array(objects[object]["dimensions"][0:3])
    dims[3:6] = 0.5 * np.array(objects[object]["dimensions"][0:3])
    new_dict[object]["dimensions"] = list(dims/100)
    new_dict[object]["scaling"] = [1,1,1]

with open('objects.json', 'w') as file:
    json.dump(new_dict, file,indent = 4)