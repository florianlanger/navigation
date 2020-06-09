import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, resnet18

class Encoder(nn.Module):
    def __init__(self,pretrained_model,encoding_dimensions,output_dim):
        super(Encoder, self).__init__()
        self.final_embedding = nn.Linear(encoding_dimensions, 10)
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128,output_dim)
        self.pretrained_model = pretrained_model

    def single_image(self,x):
        x = self.pretrained_model(x)
        x = F.relu(self.final_embedding(x))
        return x

    def forward(self,x):
        image1, image2 = x[:,0],x[:,1]
        emb_1 , emb_2 = self.single_image(image1), self.single_image(image2)
        x = torch.cat((emb_1,emb_2),dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

