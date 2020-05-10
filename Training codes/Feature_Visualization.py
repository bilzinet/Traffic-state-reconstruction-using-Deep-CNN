# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:31:58 2019

@author: NYUAD

Visualization of Speed Construction CNN Model

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2
from sklearn.model_selection import train_test_split


from keras.models import load_model
from keras import models

def load_image( subfolder) :
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
output_Y = load_image( '/Full/')
input_X = load_image( '/Sample/')

# Split data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(input_X, output_Y, test_size=0.1, random_state=13)

model = load_model('./Speed-reconstruct-cnnmodel-9.h5')
model.summary()

# --------------------------------------------------------------------------- #
# --------------------------- Activation layer outputs ---------------------- #

sample_no = 100
sample_x = y_train[sample_no,:,:,:]
plt.figure()
plt.imshow(sample_x)
plt.show()

layer_outputs = [layer.output for layer in model.layers[1:]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(sample_x.reshape(1,80,60,3))

layer_names = []
for layer in model.layers[1:]:
    layer_names.append(layer.name)
    
sample_no = 100
layer_num = 0
layer_activation = activations[layer_num]
print(layer_activation.shape)

n_cols = 16
n_rows = 8
layer_name = layer_names[layer_num]
width_x = layer_activation.shape[2]
width_y = layer_activation.shape[1]
display_grid = np.zeros((n_rows*width_y, n_cols*width_x))

for row_id, row in enumerate(range(0, display_grid.shape[0], width_y)):
    for col_id, col in enumerate(range(0, display_grid.shape[0], width_x)):
        display_grid[row:row+width_y, col:col+width_x] = layer_activation[:, :,:, row_id*n_rows+col_id]

scale = 0.05
plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.savefig(layer_name)
plt.colorbar()
plt.show()

