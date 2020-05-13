
import numpy as np
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Tensorflow imports
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Argparse import
import argparse

def process_image(image):
    '''
    The process_image function should take in an image (in the form of a NumPy array) and return an image in the form of a NumPy array with shape (224, 224, 3).
    '''
    # First, convert the image into a TensorFlow Tensor then resize it to the appropriate size using tf.image.resize
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224,224))
    #Second, the pixel values of the input images are typically encoded as integers in the range 0-255, but the model expects the pixel values to be floats in the range 0-1. 
    # #Therefore, you'll also need to normalize the pixel values
    image /= 255

    image = image.numpy()
    return image
    
def prediction(image_path,model,top_k):
    # Load the model passed into the function
    reloaded_keras_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    
    # Process the image selected
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)

    #The image returned by the process_image function is a NumPy array with shape (224, 224, 3) but the model expects the input images to be of shape (1, 224, 224, 3). This extra dimension represents the batch size.
    # Use  np.expand_dims() function to add the extra dimension.
    processed_test_image = np.expand_dims(processed_test_image, axis=0)

    # Create the model to get the probabilities by using the model and softmax layer as the input
    probability_model = tf.keras.Sequential([reloaded_keras_model, 
                                          tf.keras.layers.Softmax()])
    
    # Get the predictions by using the probability model to predict the input image 
    predictions = probability_model.predict(processed_test_image)
    
    # Get the index of the top 10 probabilities
    top_idxs = predictions[0].argsort()[-top_k:][::-1]
    
    # Get the top 10 probabilities
    top_probabilities = predictions[0][top_idxs]
    probs = top_probabilities
    
    # Get the labels (the index of the probabilities)
    labels_nums = [str(idx) for idx in top_idxs]
    classes = labels_nums
    return probs, classes


def get_label_names(json_file, labels):
    '''
    Given json_file that contains the label names for the label numbers, return the correct label names from array 'label'
    '''
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    new_class_names = {}
    for key in class_names:
        new_class_names[int(key)-1]=class_names[key]
    label_names = [new_class_names[int(i)] for i in labels]
    return label_names

# Command line application:

parser = argparse.ArgumentParser(description='Predict image label.')
# Add image path arguement
parser.add_argument('img_path', type=str,                   
                    help='the path to the image')

# Add model path arguement
parser.add_argument('my_model', type=str,                
                    help='the path to the model')

# Add optional top k probabilities arguement
parser.add_argument('--top_k', type=int,
                    dest = 'top_k',
                    default=1,
                    help='Return the top K most likely classes')

# Add optional category names of top k arguement
parser.add_argument('--category_names', type=str, 
                    dest = 'json_file',
                    default = 'label_map.json',
                    required=False,
                    help='Path to a JSON file mapping labels to flower names')

args = parser.parse_args()
# Call the prediction function to get the top k probabilities and corresponding labels
probs, classes = prediction(args.img_path, args.my_model, args.top_k)
# Print the probabilities and class numbers to the console
print(f'Proabilities: {probs}')
print(f'Label Numbers: {classes}')


# Call the get_label_names function with the JSON file with class names and the classes returned from prediction
label_names = get_label_names(args.json_file, classes)
# Print the class names to the console
print(f'Label Names: {label_names}')

