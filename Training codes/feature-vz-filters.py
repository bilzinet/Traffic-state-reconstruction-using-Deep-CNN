# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:14:32 2019

@author: btt1

Visualization of CNN generated speed maps

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.models import load_model
from keras import models
from sklearn.model_selection import train_test_split
import glob
import cv2

# --------------------------------------------------------------------------- #
# -------------------------- Loading model and data ------------------------- #

def load_image(subfolder) :
    imgsTotalNum = (glob.glob('./Data_11072019b/{}/*.jpg'.format(subfolder)))
    data_All = np.zeros((len(imgsTotalNum),h,w,3))
    for i, filepath in enumerate(imgsTotalNum):
        data = cv2.imread(filepath)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = cv2.resize(data,(w,h),interpolation = cv2.INTER_AREA)/255
        data_All[i] = data
    return data_All

# Call the input and outputs
h=80; w=60
output_Y = load_image('/Full/')
input_X = load_image('/Sample/')

# Split data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(input_X, output_Y, test_size=0.5, random_state=13)

model = load_model('./Speed-reconstruct-cnnmodel-9.h5')
model.summary()

layer_names = []
for layer in model.layers[1:]:
    layer_names.append(layer.name)
print('layer names: ', layer_names)

# --------------------------------------------------------------------------- #
from keras import backend as K
from keras.preprocessing.image import save_img

# dimensions of the generated pictures for each filter.
img_width = w
img_height = h
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def gradient_ascent(iterate):
    # step size for gradient ascent
    step = 0.7

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_height, img_width, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    num_epochs = 100
    for i in range(num_epochs):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

#        print('------>Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break
        
    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))

def build_nth_filter_loss(filter_index, layer_name):
    """
    We build a loss function that maximizes the activation
    of the nth filter of the layer considered
    """
    
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])
    
    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    return iterate

layers = ['conv2d_8', 'conv2d_9', 'conv2d_10']
kept_filters = []
filters_dict = dict()
for layer_name in layers:
    layer = model.get_layer(layer_name)
    print('Processing filter for layer:', layer_name)
    for filter_index in range(layer.output.shape[-1]):
    #    print('filter_index')
        gradient_ascent(build_nth_filter_loss(filter_index, layer_name))
    filters_dict[layer.name] = kept_filters
    kept_filters = []

for layer_name, kept_filters in filters_dict.items():
    print(layer_name, len(kept_filters))
    

def stich_filters(kept_filters, layer_name):
    # By default, we will stich the best 64 (n*n) filters on a 8 x 8 grid.
    n = int(np.sqrt(len(kept_filters)))
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((height, width, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
                height_margin: height_margin + img_height,
                width_margin: width_margin + img_width, :] = img
                    
    # save the result to disk
#    save_img('stitched_filters_{}.pdf'.format(layer_name), stitched_filters)
    plt.figure()
    plt.imshow(stitched_filters)
    plt.savefig('Stiching filters for {}'.format(layer_name))
    
    
for layer_name, kept_filters in filters_dict.items():
    print('Stiching filters for {}'.format(layer_name))
    stich_filters(kept_filters, layer_name)
    print('Completed.')