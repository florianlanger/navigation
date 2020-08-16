
import numpy as np
import torch

objects =  {
                        "arm chair": {"dimensions":[-0.7,1.1,0.46,0.6,0.6,0.46],"scaling":[1,1,1]},
                        "bookshelf": {"dimensions":[-1.91,0.67,1.74,0.14,0.95,0.35],"scaling":[1,1,1]},
                        "desk chair": {"dimensions":[-1.69,1.96,0.46,0.373,0.24,0.46],"scaling":[1,1,1]},
                        "lamp": {"dimensions":[-0.75,0.67,2.39,0.15,0.13,0.20],"scaling":[1,1,1]},
                        "sideboard": {"dimensions":[-1.78,0.67,0.42,-0.24,0.95,0.40],"scaling":[1,1,1]},
                        "sofa": {"dimensions":[1.1,0.3,0.5,0.55,0.8,0.4],"scaling":[1,1,1]},
                        "couch table": {"dimensions":[-0.87,-0.20,0.20,0.35,0.32,0.21],"scaling":[1,1,1]},
                        "big table": {"dimensions":[2.64,0.68,0.36,0.52,0.49,0.38],"scaling":[1,1,1]},
                        "radiator": {"dimensions":[-0.52,2.23,0.40,0.5,0.07,0.4],"scaling":[1,1,1]},
                        "sink": {"dimensions":[2.16,2.76,0.81,0.2,0.19,0.15],"scaling":[1,1,1]},

                    }

def objects_to_no_fly(objects):
    objects_no_fly = torch.zeros((len(objects),2,3))
    for i,key in enumerate(objects):
        objects_no_fly[i,0] = torch.from_numpy(np.array(objects[key]['dimensions'][:3]) - objects[key]['dimensions'][3:6])
        objects_no_fly[i,1] = torch.from_numpy(np.array(objects[key]['dimensions'][:3]) + objects[key]['dimensions'][3:6])
    return objects_no_fly

print(objects_to_no_fly(objects))