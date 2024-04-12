import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml
from pennylane import numpy as np
from torch.nn import functional as F

class QResNet(nn.Module):
    def __init__(self, circuit, weight_shapes, n_classes=10):
        super(QResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features

        # Replace the fully connected layer with a smaller one
        self.resnet.fc = nn.Linear(num_ftrs, 4)
        self.resnet.bn = nn.BatchNorm1d(4)
        
        # Integrate the quantum layer
        self.resnet.qn = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)

        # Final classification layer
        self.resnet.fc2 = nn.Linear(4, n_classes)

        # Printing the quantum circuit using qml.draw
        self.print_quantum_circuit(circuit, weight_shapes)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.5)
        x = self.resnet.fc(x)
        x = self.resnet.bn(x)
        x = F.leaky_relu(x)
        
        if torch.isnan(x).any():
            print("NaN detected in tensor before passing to quantum layer")
        
        x = torch.stack([self.resnet.qn(i) for i in x]).to(torch.device("mps"))
        x = self.resnet.fc2(x)
        
        return F.softmax(x, dim=1)

    def print_quantum_circuit(self, circuit, weight_shapes):
        # Create dummy data with the correct shape
        dummy_inputs = torch.randn(self.resnet.fc.in_features)
        dummy_params = torch.randn(weight_shapes['weights'])
        # Using qml.draw to generate a visual representation of the circuit
        drawer = qml.draw(circuit)
        print("\nQuantum Circuit:")
        print(drawer(dummy_inputs, dummy_params))
        print("\n")

class QResNetAMP(nn.Module):
    def __init__(self, circuit, weight_shapes, n_classes=10):
        super(QResNetAMP, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features

        # Replace the fully connected layer with a smaller one
        self.resnet.fc = nn.Linear(num_ftrs, 16)
        self.resnet.bn = nn.BatchNorm1d(16)
        
        # Integrate the quantum layer
        self.normalize = nn.functional.normalize
        self.resnet.qn = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)

        # Final classification layer
        self.resnet.fc2 = nn.Linear(4, n_classes)

        # Printing the quantum circuit using qml.draw
        self.print_quantum_circuit(circuit, weight_shapes)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.5)
        x = self.resnet.fc(x)
        x = self.resnet.bn(x)
        
        if torch.isnan(x).any():
            print("NaN detected in tensor before passing to quantum layer")
        x = torch.stack([self.resnet.qn(i) for i in x])
        x = self.resnet.fc2(x)
        
        return F.softmax(x, dim=1)

    def print_quantum_circuit(self, circuit, weight_shapes):
        # Create dummy data with the correct shape
        dummy_inputs = torch.randn(16)
        dummy_params = torch.randn(weight_shapes['weights'])
        # Using qml.draw to generate a visual representation of the circuit
        drawer = qml.draw(circuit)
        print("\nQuantum Circuit:")
        print(drawer(dummy_inputs, dummy_params))
        print("\n")