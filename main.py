import torch
from Dataset import EuroSATLoader
import time
import schedulefree as sf
from QResNet import QResNet, ResNet
from QNN4EO import QNN4ESAT
from torch.nn import functional as F
import argparse
import logging

import pennylane as qml

REPETITIONS = -1

def parse_args():
    parser = argparse.ArgumentParser(description="Train a QResNet or QResNetAMP model.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', type=str, choices=['qresnet', 'resnet', 'qnn4eo'], default='qresnet', help='Model to use for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--n_dataset_examples', type=int, default=27000, help='Number of examples in the dataset.')
    parser.add_argument('--log_outputs', type=bool, default=True, help='Whether to log outputs to a file.')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='File to log training outputs.')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer for training.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for the optimizer.')
    parser.add_argument('--test', type=bool, default=False, help='Whether to evaluate the model on the test set.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the images in the dataset.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained resnet.')
    parser.add_argument('--res_layers', type=int, default=18, help='Number of resnet layers.')
    parser.add_argument('--repetitions', type=int, default=4, help='Number of repetitions of the variational layer.')
    parser.add_argument('--reduction_layers', type=bool, default=False, help='Use reduction layers in resnets.')
    return parser.parse_args()

def test_model(val_loader, model, device, optimizer):
    model.eval()
    optimizer.zero_grad()  # Reset gradients; although typically not needed in eval
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    average_loss = total_loss / total
    accuracy = 100 * correct / total
    
    # Decorative header and footer for test results
    test_header = " Test Results ".center(80, '-')
    test_footer = "-".center(80, '-')
    test_info = f"val_acc: {accuracy:.2f}%, loss: {average_loss:.4f}".center(80)
    
    # Print formatted test results
    print(f'\n{test_header}\n{test_info}\n{test_footer}\n')
    logging.info(f'\t\tval_acc: {accuracy:.2f}%, val_loss: {average_loss:.4f}')

def train_qresnet(train_loader, val_loader, model, optimizer, device, num_epochs=10, log_file='training_log.txt'):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    model.to(device)
    logging.info("Training started")
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.train()
        start_time = time.time()
        running_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct_predictions / total_samples
        elapsed_time = time.time() - start_time
        
        # Print formatted epoch results
        epoch_header = f" Epoch {epoch + 1}/{num_epochs} ".center(80, '=')
        epoch_footer = "=".center(80, '=')
        epoch_info = f"acc: {epoch_accuracy:.4f}% - loss: {epoch_loss:.2f} - time: {elapsed_time:.0f}s".center(80)
        print(f'\n{epoch_header}\n{epoch_info}\n{epoch_footer}\n')
        
        logging.info(f"[{epoch + 1}/{num_epochs}] - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.2f}%")
        
        # Evaluate the model on the validation set after each epoch
        test_model(val_loader, model, device, optimizer)

    # Log and print the completion of training
    completion_header = " Training Completed ".center(80, '*')
    completion_footer = "*".center(80, '*')
    print(f'\n{completion_header}\n{completion_footer}\n')
    logging.info("Training completed")

if __name__ == "__main__":
    args = parse_args()
    num_classes = args.num_classes
    REPETITIONS = args.repetitions
    """TEMPORARY"""
    dev = qml.device("lightning.qubit", wires=4)
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
    
    loader = EuroSATLoader(root="EuroSAT_RGB", image_size=args.image_size, num_classes=num_classes, batch_size=args.batch_size, random_state=42, examples=args.n_dataset_examples)
    train_loader, val_loader = loader.get_loaders()
    
    device = torch.device("mps")
    
    if args.model == 'qresnet':
        model = QResNet(fully_connected_circuit, weight_shapes_fc, n_classes=num_classes, pretrained=args.pretrained, res_layers=args.res_layers)
    elif args.model == 'resnet':
        model = ResNet(n_classes=num_classes, pretrained=args.pretrained, res_layers=args.res_layers, reduction_layers=args.reduction_layers)
    else:
        model = QNN4ESAT(fully_connected_circuit, weight_shapes_fc, num_classes=num_classes)
    
    if args.optimizer == 'adam':
        optimizer = sf.AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=1e-3)
    else:
        optimizer = sf.SGDScheduleFree(model.parameters(), lr=args.lr, weight_decay=1e-4, warmup_steps=10)
    
    train_qresnet(train_loader, val_loader, model, optimizer, device, num_epochs=args.epochs, log_file=args.log_file)
    if args.test:
        test_model(val_loader, model, device)
