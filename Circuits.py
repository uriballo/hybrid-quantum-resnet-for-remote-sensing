import pennylane as qml
from pennylane import numpy as np
from main import REPETITIONS

dev = qml.device("lightning.qubit", wires=4)
weight_shapes = {"weights": (4,)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(inputs, weights):
    for j in range(4):
        qml.RY(np.pi * inputs[j], wires=j)
        qml.Hadamard(wires=j)
   
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)

    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[0, 3])
    
    qml.RY(weights[2], wires=0)
    qml.RY(weights[3], wires=3)
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

def repetition_block(W):
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
        
    
    qml.RY(W[0], wires=0)
    qml.RX(W[1], wires=0)
    qml.RZ(W[2], wires=0)
    
    qml.RY(W[3], wires=1)
    qml.RX(W[4], wires=1)
    qml.RZ(W[5], wires=1)
    
    qml.RY(W[6], wires=2)
    qml.RX(W[7], wires=2)
    qml.RZ(W[8], wires=2)
   
    qml.RY(W[9], wires=3)
    qml.RX(W[10], wires=3)
    qml.RZ(W[11], wires=3)


repetitions = REPETITIONS
weight_shapes_fc = {"weights": (repetitions, 4 * 3)}
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def fully_connected_circuit(inputs, weights):
    for i in range(4):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i], wires=i)
    
    for i in range(repetitions):
       repetition_block(weights[i])
    
    return [qml.expval(qml.PauliX(i)) for i in range(4)]    

def fully_connected_residual_circuit(inputs, weights):
    for i in range(4):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i], wires=i)
    
    for i in range(repetitions):
       repetition_block(weights[i])
    
    # apply residual connection  
    repetition_block(weights[2])
    
    return [qml.expval(qml.PauliX(i)) for i in range(4)]    

amplitude_dev = qml.device("lightning.qubit", wires=4)
amplitude_shapes = {"weights": (4,)}
@qml.qnode(amplitude_dev, interface= "torch", diff_method="adjoint")
def amplitude_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(4), normalize=True)
    for i in range(4):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i] * np.pi , wires=i)
        
    entangling_layer()
    
        
    for i in range(2):
        qml.RX(weights[i], wires=i)
        qml.RZ(weights[i+2], wires=i+2)
    
    entangling_layer()
        
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


classifier_dev = qml.device("lightning.qubit", wires=9)
classifier_shapes = {"weights": (9,)}
@qml.qnode(classifier_dev, interface="torch", diff_method="adjoint")
def classifier_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(9), normalize=True)
    #for i in range(10):
    #    qml.Hadamard(wires=i)
        
    for i in range(9):
        qml.RX(weights[i], wires=i)
        
    return [qml.expval(qml.PauliZ(i)) for i in range(9)]
