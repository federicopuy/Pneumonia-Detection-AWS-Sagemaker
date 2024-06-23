import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import boto3
import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    '''
    Function to evaluate the model on the test dataset.
    '''
    model.eval()  # Set the model to evaluation mode
    running_loss = 0
    running_corrects = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)  # Use target instead of labels
            
            _, preds = torch.max(output, 1)  # Use output instead of outputs
            
            running_loss += loss.item() * data.size(0)  # Use data instead of inputs
            running_corrects += torch.sum(preds == target.data).item()  # Use target instead of labels

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    logger.info(f" Running corrects: {running_corrects}")
    logger.info(f" Dataset size: {len(test_loader.dataset)}")
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Test Accuracy: {total_acc}")

def train(model, train_loader, criterion, optimizer, device, epochs):
    '''
    Function to train the model 
    '''
    model.train()  # Set the model to training mode
    
    total_correct_predictions = 0  # Initialize total correct predictions
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            
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
            print(f"Running total: {running_total}")
            print(f"Running correct: {running_corrects}")
        

        # Calculate the training loss and training accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * running_corrects / running_total
       
        # Log loss and accuracy for the epoch
        message = f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.2f}%, Total images: {running_total}, Total correct: {running_corrects}"
        print(message)
        logger.info(message)
    
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

def create_data_loaders(data, batch_size):
    '''
    Function to create data loaders.
    '''
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader

def main(args):
    '''
    Main function to train and test the model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")    
    logger.info(f"Running on Device {device}") 

    model=net() # Initialize the model
    model=model.to(device)
    
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

    train_loader = create_data_loaders(train_data, args.batch_size)
    val_loader = create_data_loaders(val_data, args.batch_size)

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
    train(model, train_loader, loss_criterion, optimizer, device, args.epochs)
            
    # Test the model
    test(model, val_loader, loss_criterion, device)
    
    # Save the trained model
    torch.save(model.state_dict(), args.path)
    # Train the model
    train(model, train_loader, loss_criterion, optimizer, device, args.epochs)
            
    # Test the model
    test(model, val_loader, loss_criterion, device)
    
    # Save the trained model
    torch.save(model.state_dict(), args.path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',type=float,default=0.9,  help='momentum (default: 0.9)')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (default: adam)')
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])        
    parser.add_argument("--train", type=str, required=False, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val", type=str, required=False, default=os.environ.get("SM_CHANNEL_VAL"))
    parser.add_argument("--test", type=str, required=False, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument('--path',type=str,default='model.vgg16')

    args=parser.parse_args()
    
    main(args)
