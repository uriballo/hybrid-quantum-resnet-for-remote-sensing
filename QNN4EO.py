import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pennylane.qnn as qnn


class QNN4ESAT(nn.Module):
    def __init__(self, circuit, weight_shapes, num_classes=10):
        super(QNN4ESAT, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 2 * 2, 4)
        
        self.qlayer = qnn.TorchLayer(circuit, weight_shapes)
        self.fc2 = nn.Linear(4, num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv2(x) 
        x = self.bn2(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv3(x) 
        x = self.bn3(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv4(x) 
        x = self.bn4(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)     

        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x)).to(torch.device("cpu"))
        x = torch.stack([self.qlayer(i) for i in x]).to(torch.device("mps"))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)