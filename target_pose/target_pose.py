import numpy as np
from matplotlib import pyplot as plt
import os



def draw_setup(object_box,current_pose,dir_path):
    fig = plt.figure()

    corner_1 = (object_box[0]+object_box[3],object_box[1]-object_box[4])
    corner_2 = (object_box[0]+object_box[3],object_box[1]+object_box[4])
    corner_3 = (object_box[0]-object_box[3],object_box[1]+object_box[4])
    corner_4 = (object_box[0]-object_box[3],object_box[1]-object_box[4])

    plt.plot([corner_1[0],corner_2[0]],[corner_1[1],corner_2[1]],color='blue')
    plt.plot([corner_2[0],corner_3[0]],[corner_2[1],corner_3[1]],color='blue')
    plt.plot([corner_3[0],corner_4[0]],[corner_3[1],corner_4[1]],color='blue')
    plt.plot([corner_4[0],corner_1[0]],[corner_4[1],corner_1[1]],color='blue')
    plt.xlim(-2,1)
    plt.ylim(-1,2)
    plt.scatter(current_pose[0],current_pose[1],label='Current Pose',color='green')
    plt.scatter(object_box[0],object_box[1],label='Center Object',color='red')
    plt.legend()

    
    r = object_box[:2] - current_pose[:2]

    # divide into sectors
    v1 = 10 * rotate(r,np.pi/4)
    v2 = 10 * rotate(r,3*np.pi/4)
    vs = [v1,v2]

    plt.plot([object_box[0]-v1[0],object_box[0]+v1[0]],[object_box[1]-v1[1],object_box[1]+v1[1]],'b-')
    plt.plot([object_box[0]-v2[0],object_box[0]+v2[0]],[object_box[1]-v2[1],object_box[1]+v2[1]],'b-')
    plt.gca().set_aspect('equal', adjustable='box')

    #find intersects
    sides = [(corner_1,corner_2),(corner_2,corner_3),(corner_3,corner_4),(corner_4,corner_1)]




    for v in vs:
        diagonal = (tuple(object_box[:2] - v),tuple(object_box[:2] + v))
        for side in sides:
            x1,y1 = line_intersection(diagonal,side)
            plt.scatter(x1,y1)
        
    




    fig.savefig(dir_path+'/test.png',dpi=70)
    plt.close(fig)



def rotate(vector,angle):
    x = np.cos(angle) * vector[0] - np.sin(angle) * vector[1]
    y = np.sin(angle) * vector[0] + np.cos(angle) * vector[1]
    return np.array([x,y])

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y




def main():
    object_box = np.array([-0.7,1.1,0.46,0.6,0.8,0.46])

    current_pose = np.array([1.,1.,0,])
    instruction = 'left'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    draw_setup(object_box,current_pose,dir_path)

main()

