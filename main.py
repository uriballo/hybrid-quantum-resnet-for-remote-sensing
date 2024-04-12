import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from Dataset import EuroSATLoader
import time
import schedulefree as sf
from QResNet import QResNet, QResNetAMP
import argparse
from Circuits import circuit, weight_shapes
from Circuits import amplitude_circuit, amplitude_shapes

def parse_args():
    parser = argparse.ArgumentParser(description="Train a QResNet or QResNetAMP model.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', type=str, choices=['qresnet', 'qresnetAMP'], default='qresnet', help='Model to use for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--test_set_size', type=int, default=1000, help='Size of the test dataset.')
    parser.add_argument('--n_dataset_examples', type=int, default=27000, help='Number of examples in the dataset.')
    parser.add_argument('--log_outputs', type=bool, default=False, help='Whether to log outputs to a file.')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='File to log training outputs.')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer for training.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for the optimizer.')
    parser.add_argument('--test', type=bool, default=False, help='Whether to evaluate the model on the test set.')
    return parser.parse_args()

def train_qresnet(train_loader, val_loader, model, optimizer, criterion, device, num_epochs=10):
    model.to(device)
    print("Training started".center(30, '-'))
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct_predictions / total_samples
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs}".ljust(30))
        print(f"Loss: {epoch_loss:.4f}".ljust(30))
        print(f"Accuracy: {epoch_accuracy:.2f}%".ljust(30))
        print(f"Time: {elapsed_time:.0f}s".ljust(30))
        print('-' * 30)
    print("Training completed".center(30, '-'))

if __name__ == "__main__":
    args = parse_args()
    
    loader = EuroSATLoader(root="EuroSAT_RGB", image_size=256, batch_size=args.batch_size, test_size=0.2, random_state=42, examples=args.n_dataset_examples)
    print("Loading data...")
    train_loader, val_loader = loader.get_loaders()
    
    if args.model == 'qresnet':
        model = QResNet(circuit, weight_shapes, n_classes=10)
        device = torch.device("mps")
    else:
        model = QResNetAMP(amplitude_circuit, amplitude_shapes, n_classes=10)
        device = torch.device("cpu")
    
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = sf.AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        optimizer = sf.SGDScheduleFree(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    train_qresnet(train_loader, val_loader, model, optimizer, criterion, device, num_epochs=args.epochs)
