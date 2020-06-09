import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, number_outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,10, kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.0)
        self.fc1 = nn.Linear(9680, 50)
        self.fc2 = nn.Linear(50, 7)
        # as concatenate two rows of 7
        self.fc3 = nn.Linear(14, 128)
        self.fc4 = nn.Linear(128,number_outputs)

    def single_image(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training,p=0.0)
        x = F.relu(self.fc2(x))
        return x

    def forward(self,x):
        image1, image2 = x[:,0],x[:,1]
        emb_1 , emb_2 = self.single_image(image1), self.single_image(image2)
        x = torch.cat((emb_1,emb_2),dim=1)
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
