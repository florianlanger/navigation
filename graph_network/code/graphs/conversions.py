import torch

class Converter(object):
    def __init__(self,min_position,max_position,steps,number_poses,corners_no_fly_zone=None):
        self.min_position = min_position
        self.max_position = max_position
        self.steps = steps
        self.move_to_coords = self.initialise_move_to_coords()
        self.number_poses = number_poses
        self.number_dims = min_position.shape[0]
        self.corners_no_fly_zone = corners_no_fly_zone
        self.number_poses_each_dim = torch.round((self.max_position - self.min_position)/self.steps)+1 #.cuda() + 1
        if self.number_dims == 4:
            self.prod_number_poses = torch.tensor([torch.prod(self.number_poses_each_dim[0:]),torch.prod(self.number_poses_each_dim[1:]),
            torch.prod(self.number_poses_each_dim[2:]),torch.prod(self.number_poses_each_dim[3:]),1.]) #.cuda()
        elif self.number_dims == 3:
            self.prod_number_poses = torch.tensor([torch.prod(self.number_poses_each_dim[0:]),torch.prod(self.number_poses_each_dim[1:]),
            torch.prod(self.number_poses_each_dim[2:]),1.]) #.cuda()
    

    def index_to_pose(self,index):
        if index < self.number_poses and index >=0:
            index_each_direction = index * torch.ones(self.number_dims) #.cuda()
            index_each_direction = (index_each_direction % self.prod_number_poses[:self.number_dims])// self.prod_number_poses[1:]
            pose = self.min_position + self.steps * index_each_direction
            return pose
        else:
            raise Exception('Not a valid index')

    def pose_to_index(self,pose):
        indices = torch.round((pose - self.min_position)/self.steps).to(torch.float)
        index = torch.dot(indices,self.prod_number_poses[1:])
        if index < self.number_poses and index >= 0:
            return int(index.item())
        else:
            raise Exception('Not a valid pose')

    def validate_pose(self,pose):
        if torch.all(pose[:3] + 0.0001 >= self.min_position[:3]) and torch.all(pose[:3] - 0.0001 <= self.max_position[:3]):
            return True
        else:
            return False

    def check_flyable_index(self,index):
        pose = self.index_to_pose(index)
        return self.check_flyable_pose(pose)

    def check_flyable_pose(self,pose):
        if pose.shape[0] == 3:
            return True
        elif pose.shape[0] == 4:
            in_no_fly_1 = (torch.all(pose[:3] + 0.0001 >= self.corners_no_fly_zone[0,0]) and torch.all(pose[:3] - 0.0001<= self.corners_no_fly_zone[0,1]))
            in_no_fly_2 = (torch.all(pose[:3] + 0.0001 >= self.corners_no_fly_zone[1,0]) and torch.all(pose[:3] - 0.0001<= self.corners_no_fly_zone[1,1]))
            if (in_no_fly_1 or in_no_fly_2):
                return False
            else:
                return True
    

    def map_pose(self,pose):
        if pose.shape[0] == 3:
            return pose
        elif pose.shape[0] == 4:
            pose[3] = pose[3] % 1.
            return pose

    def initialise_move_to_coords(self):
        zeros = torch.tensor([0.,0.,0.,0.]) #.cuda()
        stay = zeros.clone()
        term = zeros.clone()

        pos_x = zeros.clone()
        pos_x[0] = self.steps[0]
        neg_x = zeros.clone()
        neg_x[0] = - self.steps[0]

        pos_y = zeros.clone()
        pos_y[1] = self.steps[1]
        neg_y = zeros.clone()
        neg_y[1] = - self.steps[1]

        pos_z = zeros.clone()
        pos_z[2] = self.steps[2]
        neg_z = zeros.clone()
        neg_z[2] = - self.steps[2]

        pos_rot = zeros.clone()
        pos_rot[3] = self.steps[3]
        neg_rot = zeros.clone()
        neg_rot[3] = - self.steps[3]

        return {0:stay,1:pos_x,2:neg_x,3:pos_y,4:neg_y,5:pos_z,6:neg_z,7:pos_rot,8:neg_rot,9:term} 


# index = 6533
# min_pos = torch.tensor([-1.3,-0.5,0.2,0.])
# max_pos = torch.tensor([1.8,1.4,1.7,270.])
# steps = torch.tensor([0.1,0.1,0.1,90.])
# corners_no_fly_zone = torch.tensor([[0.5,-0.5,0.1,0.0],[1.7,1.1,0.9,270.]])
# converter = Converter(min_pos,max_pos,steps,40960,corners_no_fly_zone)
# print(converter.move_to_coords[3],converter.move_to_coords[2],converter.move_to_coords[9],converter.move_to_coords)
# pose = converter.index_to_pose(index)
# print(converter.check_flyable_index(31247))
# print(converter.check_flyable_index(31284))
# print(converter.check_flyable_pose(torch.tensor([1.1,0.3,0.7,90.])))
# print(converter.check_flyable_pose(torch.tensor([1.7,1.1,0.9,270.])))