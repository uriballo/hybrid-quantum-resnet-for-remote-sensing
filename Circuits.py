import pennylane as qml
from pennylane import numpy as np
import torch

def entangling_layer():
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    
dev = qml.device("lightning.qubit", wires=4)
weight_shapes = {"weights": (4,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(inputs, weights):
    for i in range(4):
        #qml.Hadamard(wires=i)
        qml.RY(inputs[i] * np.pi , wires=i)
    
    #entangling_layer()
    
    for i in range(2):
        qml.RX(weights[i], wires=i)
        qml.RZ(weights[i+2], wires=i+2)
    
    #entangling_layer()
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


amplitude_dev = qml.device("lightning.qubit", wires=4)
amplitude_shapes = {"weights": (4,)}
@qml.qnode(amplitude_dev, interface= "torch", diff_method="adjoint")
def amplitude_circuit(inputs, weights):
    print(inputs)
    qml.AmplitudeEmbedding(inputs, wires=range(4), normalize=True)
    for i in range(4):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i] * np.pi , wires=i)
        
    for i in range(2):
        qml.RX(weights[i], wires=i)
        qml.RZ(weights[i+2], wires=i+2)
    
    entangling_layer()
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
