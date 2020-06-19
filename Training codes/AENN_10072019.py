# ----------------- Import Libraries --------------- #

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import glob
import random

# ------------- CNN Reconstruction model ------------ #

# Input size (space-time discretization)
h=80
w=60
input_img = Input(shape=(h,w,3))

# Encoder model
x = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 3), padding='same')(x)
x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder model
x = Conv2D(16, (5, 5), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (7, 7), activation='relu', padding='same')(x)
x = UpSampling2D((2, 3))(x)
decoded = Conv2D(3, (7, 7), activation='sigmoid', padding='same')(x)

# Reconstruction model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['mse','mae'])
autoencoder.summary()

# -------------- Loading training data -------------- #

# Read in the data (images) according to the sizes
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
output_Y = load_image( '/Full/')
input_X = load_image( '/Sample/')

# Split data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(input_X,
                                                 output_Y, 
                                                 test_size=0.1, 
                                                 random_state=13)

# -------------- Train the model --------------- #

# Learning step by step
loss = []; val_loss = []; mse = []; mae = []; val_mse = []; val_mae = []
for time in range(10):
    if time == 0:
        history = autoencoder.fit(x_train, y_train, epochs=50, batch_size=1, shuffle=True, 
                        validation_data=(x_test, y_test), verbose=1)
        loss = loss+history.history['loss']
        val_loss = val_loss+history.history['val_loss']
        mse = mse+history.history['mean_squared_error']
        mae = mae+history.history['mean_absolute_error']
        val_mse = val_mse+history.history['val_mean_squared_error']
        val_mae = val_mae+history.history['val_mean_absolute_error']
        autoencoder.save('Speed-reconstruct-cnnmodel-{}.h5'.format(time))
        del autoencoder
    else:
        autoencoder = load_model('Speed-reconstruct-cnnmodel-{}.h5'.format(time-1))
        history = autoencoder.fit(x_train, y_train, epochs=50, batch_size=1, shuffle=True, 
                        validation_data=(x_test, y_test),verbose=1)
        loss = loss+history.history['loss']
        val_loss = val_loss+history.history['val_loss']
        mse = mse+history.history['mean_squared_error']
        mae = mae+history.history['mean_absolute_error']
        val_mse = val_mse+history.history['val_mean_squared_error']
        val_mae = val_mae+history.history['val_mean_absolute_error']
        autoencoder.save('Speed-reconstruct-cnnmodel-{}.h5'.format(time))
        del autoencoder

np.save('Loss_values.npy', np.array(loss))
np.save('Val_loss_values.npy', np.array(val_loss))
np.save('MSE_values.npy', np.array(mse))
np.save('Val_MSE_values.npy', np.array(val_mse))    
np.save('MAE_values.npy', np.array(mae))
np.save('Val_MAE_values.npy', np.array(val_mae))


# ------------ Model performance ------------ #

plt.figure(figsize=(4,4))
plt.plot(loss, label='Loss_train')
plt.plot(val_loss, label='Loss_test')
plt.xlabel('Number of training epochs', fontsize=10)
plt.ylabel('Binary Cross Entropy Loss function', fontsize=10)
plt.legend(fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid()
plt.savefig('Performance_plot_loss_function.pdf')

plt.figure(figsize=(4,4))
plt.plot(mse[::10], label='MSE_train')
plt.plot(val_mse[::10], label='MSE_test')
plt.xlabel('Number of training epochs', fontsize=10)
plt.ylabel('Mean squared error', fontsize=10)
plt.legend(fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid()
plt.savefig('Performance_plot_mse.pdf')

plt.figure(figsize=(4,4))
plt.plot(mae[::10], label='MAE_train')
plt.plot(val_mae[::10], label='MAE_test')
plt.xlabel('Number of training epochs', fontsize=10)
plt.ylabel('Mean absolute error', fontsize=10)
plt.legend(fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid()
plt.savefig('Performance_plot_mae.pdf')


# ---------- Test the reconstruction model ----------- #

autoencoder = load_model('Speed-reconstruct-cnnmodel-{}.h5'.format(time))
decoded_imgs = autoencoder.predict(x_test)

nn = np.arange(0,len(x_test),1)
n = random.choices(nn,k=4)

for i, j in enumerate(n):
    # display original (Full)
    ax = plt.subplot(3, len(n), i + 1)
    plt.imshow(y_test[j].reshape(h,w,3))
    plt.gray()
    ax.set_axis_off()
    
    # display original (Given Sample)
    ax = plt.subplot(3, len(n), i  + len(n) + 1)
    plt.imshow(x_test[j].reshape(h,w,3))
    plt.gray()
    ax.set_axis_off()
    
    # display reconstruction
    ax = plt.subplot(3, len(n), i + 2*len(n) + 1)
    plt.imshow(decoded_imgs[j].reshape(h,w,3))
    plt.gray()
    ax.set_axis_off()
plt.show()

from keras.utils import plot_model
plot_model(autoencoder, to_file='model.png')

filled = []
for i in range(x_train.shape[0]):
    filled.append(np.where(x_train[i].reshape(-1) != 1)[0].shape[0])
filled = np.array(filled)/(80*60*3)
plt.hist(filled, bins=10, rwidth=0.8)
#K.clear_session()
