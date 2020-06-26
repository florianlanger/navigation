import numpy as np
from matplotlib import pyplot as plt
import os

def draw_object(object_box,color,label):
    corner_1 = (object_box[0]+object_box[3],object_box[1]-object_box[4])
    corner_2 = (object_box[0]+object_box[3],object_box[1]+object_box[4])
    corner_3 = (object_box[0]-object_box[3],object_box[1]+object_box[4])
    corner_4 = (object_box[0]-object_box[3],object_box[1]-object_box[4])

    plt.plot([corner_1[0],corner_2[0]],[corner_1[1],corner_2[1]],color=color,label=label)
    plt.plot([corner_2[0],corner_3[0]],[corner_2[1],corner_3[1]],color=color)
    plt.plot([corner_3[0],corner_4[0]],[corner_3[1],corner_4[1]],color=color)
    plt.plot([corner_4[0],corner_1[0]],[corner_4[1],corner_1[1]],color=color)
    plt.scatter(object_box[0],object_box[1],color=color)


def find_point(objects,current_pose,constraints,space_dim,dir_path=None,counter=None):
    fig = plt.figure()
    for single_object,color,name in zip([objects[key] for key in objects],['red','blue'],['armchair','sofa']):
        draw_object(single_object,color,name)
    plt.xlim(space_dim[0])
    plt.ylim(space_dim[1])
    plt.scatter(current_pose[0],current_pose[1],label='Current Pose',color='green')

    plt.gca().set_aspect('equal', adjustable='box')

    all_constraints_fullfilled = False
    tries = 0
    while not all_constraints_fullfilled:
        tries +=1
        if tries == 1000:
            break
        test_x = space_dim[0][0] + np.random.random() * (space_dim[0][1] - space_dim[0][0])
        test_y = space_dim[1][0] + np.random.random() * (space_dim[1][1] - space_dim[1][0])
        test_z = space_dim[2][0] + np.random.random() * (space_dim[2][1] - space_dim[2][0])
        for i,constraint in enumerate(constraints):
            checked_constraint = check_constraint(constraint,(test_x,test_y,test_z),current_pose,objects)
            #print(i, '  ' , checked_constraint)
            if checked_constraint == False:
                break
            elif i == len(constraints) - 1:
                all_constraints_fullfilled = True

    if tries == 30000:
        raise Exception('Could not find pose that matches all the constraints {}'.format(constraints)) 
    else:

        # fig = plt.figure()
        # for single_object,color,name in zip([objects[key] for key in objects],['red','blue'],['armchair','sofa']):
        #     draw_object(single_object,color,name)
        # plt.xlim(space_dim[0])
        # plt.ylim(space_dim[1])
        # plt.scatter(current_pose[0],current_pose[1],label='Current Pose',color='green')

        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.scatter(test_x,test_y,label='Pose {}'.format(counter),color='green')
        # plt.legend()
        # fig.savefig(dir_path+'/test_{}.png'.format(counter),dpi=70)
        # plt.close(fig)
        return [test_x,test_y,test_z]

def check_constraint(constraint,point,current_pose,objects):

    object_box = objects[constraint['object_name']]

    r = object_box[:2] - current_pose[:2]

    if constraint['preposition'] == 'above':
        if point[2] > object_box[2] + object_box[5]:
            return True
        else:
            return False
    elif constraint['preposition'] == 'below':
        if point[2] < object_box[2] - object_box[5]:
            return True
        else:
            return False
    
    else:

        # divide into sectors
        v1 = 10 * rotate(r,-np.pi/4)
        v2 = 10 * rotate(r,np.pi/4)
        vs = [v1,v2]

        diagonal_up = ((object_box[0]-v1[0],object_box[1]-v1[1]),(object_box[0]+v1[0],object_box[1]+v1[1]))
        diagonal_down = ((object_box[0]-v2[0],object_box[1]-v2[1]),(object_box[0]+v2[0],object_box[1]+v2[1]))

        line_pose_to_test = ((current_pose[0],current_pose[1]),point)

        intersect_diag_up = line_intersection(diagonal_up,line_pose_to_test)
        intersect_diag_down = line_intersection(diagonal_down,line_pose_to_test)
        crossing_diag_up = check_crossing(intersect_diag_up,line_pose_to_test)
        crossing_diag_down = check_crossing(intersect_diag_down,line_pose_to_test)


        if (find_test_preposition(crossing_diag_up,crossing_diag_down) == constraint['preposition'] \
            and not in_object(objects,constraint['object_name'],point)):
            return True
        else:
            return False

    # plt.plot([diagonal_up[0][0],diagonal_up[1][0]],[diagonal_up[0][1],diagonal_up[1][1]],label='diagonal_up')
    # plt.plot([diagonal_down[0][0],diagonal_down[1][0]],[diagonal_down[0][1],diagonal_down[1][1]],label='diagonal_down')



def find_test_preposition(crossing_diag_up,crossing_diag_down):
    if crossing_diag_up and crossing_diag_down:
        test_preposition = 'behind'
    elif not crossing_diag_up and crossing_diag_down:
        test_preposition = 'right'
    elif crossing_diag_up and not crossing_diag_down:
        test_preposition = 'left'
    elif not crossing_diag_up and not crossing_diag_down:
        test_preposition = 'front'
    return test_preposition

def in_object(objects,object_name,point):
    array = objects[object_name]
    between_x = point[0] < array[0] + array[3] and point[0] > array[0] - array[3]
    between_y = point[1] < array[1] + array[4] and point[1] > array[1] - array[4]
    return (between_x and between_y)

def check_crossing(intersect,line):
    if intersect[0] > min(line[0][0],line[1][0]) and intersect[0] < max(line[0][0],line[1][0]) and intersect[1] > min(line[0][1],line[1][1]) and intersect[1] < max(line[0][1],line[1][1]):
        return True
    else:
        return False

    #sides = [(corner_1,corner_2),(corner_2,corner_3),(corner_3,corner_4),(corner_4,corner_1)]

    #diagonal_up = 

    # for v in vs:
    #     diagonal = (tuple(object_box[:2] - v),tuple(object_box[:2] + v))
    #     for side in sides:
    #         x1,y1 = line_intersection(diagonal,side)
    #         plt.scatter(x1,y1)



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


def filter_text(text_description):
    if text_description == 'keep going':
        return 'no constraint',None
    else:
        if 'armchair' in text_description:
            anchor_object = 'armchair'
        elif 'sofa' in text_description:
            anchor_object = 'sofa'
        if 'left' in text_description:
            preposition = 'left'
        elif 'right' in text_description:
            preposition = 'right'
        elif 'above' in text_description:
            preposition = 'above'
        elif 'below' in text_description:
            preposition = 'below'
        elif 'front' in text_description:
            preposition = 'front'
        elif 'behind' in text_description:
            preposition = 'behind'
        return anchor_object,preposition  
        


def main():

    objects = {'armchair':np.array([-0.7,1.1,0.46,0.6,0.8,0.46]),
                'sofa': np.array([1.1,0.3,0.5,0.55,0.8,0.4])}

    current_pose = np.array([-1,-0.5,0])
    space_dim = ((-2,2),(-1,2),(0,1.6))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    constraints = []
    for i in range(2):
        while True:
            try:
                text_instruction = input('Describe the pose: ')
                object_name,preposition = filter_text(text_instruction)
                constraints.append({'object_name':object_name,'preposition':preposition})
            except NameError:
                print("Please enter a valid description. Must contain sofa or armchair and left, right, front, behind, above or below.")
                continue
            else:
                break
    
        find_point(objects,current_pose,constraints,space_dim,dir_path,i)

#main()

