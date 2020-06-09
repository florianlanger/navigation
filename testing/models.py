import torch
import torch.nn as nn

class Combi_Model(nn.Module):
    def __init__(self,pose_model,action_predictor_model):
        super(Combi_Model, self).__init__()
        self.pose_model = pose_model
        self.action_predictor_model = action_predictor_model


    def forward(self,x):
        image1, image2 = x[:,0],x[:,1]
        pose_1 , pose_2 = self.pose_model(image1).cpu(), self.pose_model(image2).cpu()
        pose_1[:,3] = pose_1[:,3] % 1.
        pose_2[:,3] = pose_2[:,3] % 1.
        poses = torch.cat((pose_1,pose_2),dim=1)
        action_predictions = self.action_predictor_model(poses)
        print('done with model')
        return poses,action_predictions