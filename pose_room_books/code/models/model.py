import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18

class Pose_Model(nn.Module):
    def __init__(self,pretrained_model,encoding_dimensions):
        super(Pose_Model, self).__init__()
        self.fc1 = nn.Linear(encoding_dimensions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,7)
        self.pretrained_model = pretrained_model

    def forward(self,x):
        x = self.pretrained_model(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x