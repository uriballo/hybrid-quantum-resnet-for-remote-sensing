import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml
from pennylane import numpy as np
from torch.nn import functional as F

class QResNet(nn.Module):
    def __init__(self, circuit, weight_shapes, n_classes=10, pretrained=False, res_layers=18):
        super(QResNet, self).__init__()
        
        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None
        
        if res_layers == 18:
            self.resnet = models.resnet18(weights=weights)
        elif res_layers == 34:
            self.resnet = models.resnet34(weights=weights)
        else:
            self.resnet = models.resnet50(weights=weights)
            
        num_ftrs = self.resnet.fc.in_features
        
        self.fc_reduction = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )

        self.resnet.fc = nn.Linear(num_ftrs, 4)
        self.resnet.qn = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        
        self.fc_expansion = nn.Sequential(
            nn.Linear(4, 10),
        )

        self.print_quantum_circuit(circuit, weight_shapes)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc_reduction(x).to(torch.device("cpu"))
        
        if torch.isnan(x).any():
            print("NaN detected in tensor before passing to quantum layer")
        
        x = torch.stack([self.resnet.qn(i) for i in x]).to(torch.device("mps"))
        x = self.fc_expansion(x)
        
        return F.softmax(x, dim=1)

    def print_quantum_circuit(self, circuit, weight_shapes):
        dummy_inputs = torch.randn(self.resnet.fc.in_features)
        dummy_params = torch.randn(weight_shapes['weights'])
        drawer = qml.draw(circuit)
        print("\nQuantum Circuit:")
        print(drawer(dummy_inputs, dummy_params))
        print("\n")

class ResNet(nn.Module):
    def __init__(self, res_layers=18, reduction_layers=True, n_classes=10, pretrained=False):
        super(ResNet, self).__init__()

        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None
        
        if res_layers == 18:
            self.resnet = models.resnet18(weights=weights)
        elif res_layers == 34:
            self.resnet = models.resnet34(weights=weights)
        else:
            self.resnet = models.resnet50(weights=weights)

        num_ftrs = self.resnet.fc.in_features

        if reduction_layers:
            self.fc_reduction = nn.Sequential(
                nn.Linear(num_ftrs, 4),
                nn.BatchNorm1d(4),
                nn.ReLU(),
                nn.Linear(4, 4),
                nn.BatchNorm1d(4),
                nn.ReLU()
            )
            final_input_features = 4
        else:
            self.fc_reduction = None
            final_input_features = num_ftrs
        
        self.final_fc = nn.Linear(final_input_features, n_classes)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        
        if self.fc_reduction:
            x = self.fc_reduction(x)
        
        x = self.final_fc(x)
        
        return F.softmax(x, dim=1)