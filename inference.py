import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

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

def model_fn(model_dir):
    logger.info("In model_fn. Model directory is -")
    logger.info(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    
    model_path = os.path.join(model_dir, "model.vgg16_best_hyperparameters")
    with open(model_path, "rb") as f:
        logger.info("Loading the model")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info('MODEL-LOADED')
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')

    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    
    if content_type == JSON_CONTENT_TYPE:
        request = json.loads(request_body)
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    logger.info('In predict fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Handle grayscale images        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    logger.info("Transforming input")
    input_object = test_transform(input_object)
    logger.info("Input transformed to tensor with shape: {}".format(input_object.shape))
    input_object = input_object.unsqueeze(0)
    logger.info("Input unsqueezed to batch dimension with shape: {}".format(input_object.shape))
    input_object = input_object.to(device)
    logger.info("Input moved to device")

    logger.info("Input tensor dtype: {}".format(input_object.dtype))
    
    with torch.no_grad():
        logger.info("Calling model")
        output = model(input_object)
        logger.info("Model output obtained")
        _, predicted = torch.max(output, 1)
    
    labels = ['NORMAL', 'PNEUMONIA']
    prediction_label = labels[predicted.item()]
    
    return {'prediction': prediction_label}

def output_fn(prediction, content_type):
    logger.info('Serializing the output.')
    return json.dumps(prediction)            
