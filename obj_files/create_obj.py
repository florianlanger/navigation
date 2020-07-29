import numpy as np
import os

objects = {
                'big cupboard': {'dimensions':np.array([0.31,3.03,1.,0.31,0.61,1.]),'scaling':np.array([1.,1.,1.])},
                'sideboard': {'dimensions':np.array([0.6,0.3,0.38,0.6,0.3,0.38]),'scaling':np.array([1.,1.,1.])},
                'table': {'dimensions':np.array([2.1,0.45,0.75,0.9,0.45,0.1]),'scaling':np.array([1.,1.,1.])},
                'couch': {'dimensions':np.array([3.16,3.22,0.35,0.78,0.44,0.35]),'scaling':np.array([1.,1.,1.])},
                'stool': {'dimensions':np.array([2.68,2.38,0.2,0.3,0.2,0.2]),'scaling':np.array([1.,1.,1.])},
                'small cupboard': {'dimensions':np.array([3.93,2.42,0.35,0.17,0.22,0.35]),'scaling':np.array([1.,1.,1.])},
                'printer': {'dimensions':np.array([3.65,1.5,0.2,0.25,0.25,0.2]),'scaling':np.array([1.,1.,1.])},
                'lamp': {'dimensions':np.array([2.4,0.42,1.1,0.12,0.42,0.35]),'scaling':np.array([1.,1.,1.])}
                    }


def create_obj_single_object(object_name,dims):
    text = '\n\ng ' + object_name + '\n\n'
    text += 'v ' + str(dims[0] - dims[3]) + ' ' + str(dims[1] - dims[4]) + ' ' + str(dims[2] - dims[5]) + '\n'
    text += 'v ' + str(dims[0] - dims[3]) + ' ' + str(dims[1] - dims[4]) + ' ' + str(dims[2] + dims[5]) + '\n'
    text += 'v ' + str(dims[0] - dims[3]) + ' ' + str(dims[1] + dims[4]) + ' ' + str(dims[2] - dims[5]) + '\n'
    text += 'v ' + str(dims[0] - dims[3]) + ' ' + str(dims[1] + dims[4]) + ' ' + str(dims[2] + dims[5]) + '\n'
    text += 'v ' + str(dims[0] + dims[3]) + ' ' + str(dims[1] - dims[4]) + ' ' + str(dims[2] - dims[5]) + '\n'
    text += 'v ' + str(dims[0] + dims[3]) + ' ' + str(dims[1] - dims[4]) + ' ' + str(dims[2] + dims[5]) + '\n'
    text += 'v ' + str(dims[0] + dims[3]) + ' ' + str(dims[1] + dims[4]) + ' ' + str(dims[2] - dims[5]) + '\n'
    text += 'v ' + str(dims[0] + dims[3]) + ' ' + str(dims[1] + dims[4]) + ' ' + str(dims[2] + dims[5]) + '\n'
    
    text += "\nvn  0.0  0.0  1.0\nvn  0.0  0.0 -1.0\nvn  0.0  1.0  0.0\nvn  0.0 -1.0  0.0\nvn  1.0  0.0  0.0\nvn -1.0  0.0  0.0\n\nf  1//2  7//2  5//2\nf  1//2  3//2  7//2\nf  1//6  4//6  3//6\nf  1//6  2//6  4//6\nf  3//3  8//3  7//3\nf  3//3  4//3  8//3\nf  5//5  7//5  8//5\nf  5//5  8//5  6//5\nf  1//4  5//4  6//4\nf  1//4  6//4  2//4 \nf  2//1  6//1  8//1\nf  2//1  8//1  4//1"
    return text

dir_path = os.path.dirname(os.path.realpath(__file__)) 

with open(dir_path +'/test.obj','w') as file:
    for key in objects: 
        with open(dir_path +'/{}.obj'.format(key),'w') as file:
            text = create_obj_single_object(key,objects[key]['dimensions'])
            file.write(text)