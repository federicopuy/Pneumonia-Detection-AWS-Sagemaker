import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import os
import sys
import argparse
import logging
import time

import smdebug.pytorch as smd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def test(model, test_loader, criterion, device, hook):
    '''
    Function to evaluate the model on the test dataset.
    '''
    hook.set_mode(smd.modes.EVAL)
    model.eval()  # Set the model to evaluation mode
    running_loss = 0
    running_corrects = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            start_time = time.time()
            
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            _, preds = torch.max(output, 1)
            
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data).item()

            end_time = time.time()
            batch_time = end_time - start_time
            if batch_time > 10:  # Assuming 10 seconds as a timeout threshold for a batch
                logger.warning(f"Batch processing time is too long: {batch_time:.2f} seconds")

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, hook):
    '''
    Function to train the model 
    '''
    model.train()  # Set the model to training mode
    early_stopping = EarlyStopping(patience=5, verbose=True)  # Initialize EarlyStopping

    hook.set_mode(smd.modes.TRAIN)

    total_correct_predictions = 0  # Initialize total correct predictions
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            start_time = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()

            end_time = time.time()
            batch_time = end_time - start_time
            if batch_time > 10:  # Assuming 10 seconds as a timeout threshold for a batch
                logger.warning(f"Batch processing time is too long: {batch_time:.2f} seconds")

        # Calculate the training loss and training accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * running_corrects / running_total

        # Log loss and accuracy for the epoch
        message = f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.2f}%, Total images: {running_total}, Total correct: {running_corrects}"
        logger.info(message)

        # Validate the model
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        logger.info(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Check if early stopping criteria are met
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    
    # Load the best model state
    model.load_state_dict(torch.load('checkpoint.pt'))

def validate(model, val_loader, criterion, device):
    '''
    Function to validate the model on the validation dataset.
    '''
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    
    return val_loss, val_accuracy

def net():
    '''
    Function to initialize the model.
    '''
    model = models.vgg16(pretrained=True)  # Load pre-trained VGG16 model

    for param in model.parameters():
        param.requires_grad = False
        
    # Get the number of input features to the classifier
    num_features = model.classifier[6].in_features
    
    # Modify the final fully connected layer for binary classification
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 256),  # Add a new fully connected layer
        nn.ReLU(inplace=True),        # ReLU activation
        nn.Linear(256, 2),             # Change the number of output neurons to 2
    )
    return model
    
def create_data_loaders(data, batch_size, shuffle=True, num_workers=4):
    '''
    Function to create data loaders.
    '''
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def main(args):
    '''
    Main function to train and test the model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")    
    logger.info(f"Running on Device {device}") 

    model = net()  # Initialize the model
    model = model.to(device)

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    
    print('Finished getting the base model')
    logger.info('Finished getting the base model')

    logger.info(f'hyperparams: {args}')

    # Create data loaders
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = ImageFolder(root=args.train, transform=transform)
    val_data = ImageFolder(root=args.val, transform=transform)

    train_loader = create_data_loaders(train_data, args.batch_size, num_workers=2)
    val_loader = create_data_loaders(val_data, args.batch_size, shuffle=False, num_workers=2) # We disable shuffling to ensure reproducibility.

    # Create loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()

    optimizer_name = args.optimizer.lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError("Invalid optimizer name. Supported options are 'adam' and 'sgd'.")
        
    # Train the model
    train(model, train_loader, val_loader, loss_criterion, optimizer, device, args.epochs, hook)
            
    # Test the model
    test(model, val_loader, loss_criterion, device, hook)
    
    # Save the trained model
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, args.path))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',type=float,default=0.9,  help='momentum (default: 0.9)')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (default: adam)')
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])        
    parser.add_argument("--train", type=str, required=False, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val", type=str, required=False, default=os.environ.get("SM_CHANNEL_VAL"))
    parser.add_argument("--test", type=str, required=False, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument('--path',type=str,default='model.vgg16_best_hyperparameters')

    args=parser.parse_args()
    
    main(args)
