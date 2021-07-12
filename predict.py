# import libraries 
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json


# extract an image
def get_image(im):
    im = Image.open(im)
    im = np.asarray(im)
    processed_im = process_image(im)
    im = np.expand_dims(processed_im, axis= 0)
    return im

# return an image in the form of a NumPy array with shape (224, 224, 3)
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image,(224,224))
    image = tf.cast(image, tf.float32)
    image /= 255
    return image.numpy()

# predict the flower's label
def predict(image, model, top_k):
    predict = model.predict(image)
    result = tf.math.top_k(predict, top_k)
    prob = result.values.numpy()
    classes = result.indices.numpy() + 1
    return prob, classes

# mapping from label to category name
def get_names(file, classes):
    name = []
    with open(file, 'r') as f:
        class_names = json.load(f)
        
    for i in classes:
        name.append(class_names[i.astype(str)])
    return name
    
    
if __name__ == '__main__':
    #initialize the parser
    parser = argparse.ArgumentParser(
        description= 'Part 2.. Building the Command Line Application'
    )
    
    #add the parameter 
    parser.add_argument('im', help='Image Path') #positional
    parser.add_argument('model', help='model') #positional
    parser.add_argument('--top_k', help='Top classes probability', type = int, default = 5) #optional
    parser.add_argument('--category_names', help='Flower Name') #optional
    
    #parse the arguments
    args = parser.parse_args()
    
    
    # predict the classes
    image = get_image(args.im)
    model = tf.keras.models.load_model(args.model, custom_objects= {'KerasLayer':hub.KerasLayer}, compile = False)
    probs, classes = predict(image, model, args.top_k)
    print('probs: ', probs, 'classes: ', classes)
    
    if args.category_names != None:
        names = get_names(args.category_names, classes[0])
        print('probs: ', probs, 'Flowers classes: ', names)