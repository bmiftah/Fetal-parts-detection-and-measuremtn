import config

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

K.set_image_data_format('channels_last')
from UNet.unet import BatchActivate, convolution_block, residual_block, build_model
from metrics import dice_coef, dice_coef_loss

# data generator
def Generator(X_list, y_list, batch_size = 16):
    c = 0

    while(True):
        X = np.empty((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
        y = np.empty((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
        
        for i in range(c,c+batch_size):
            image = X_list[i]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask =  y_list[i]
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
    
            X[i - c] = image
            y[i - c] = mask
        
        X = X[:,:,:,np.newaxis] / 255
        y = y[:,:,:,np.newaxis] / 255
        
        c += batch_size
        if(c+batch_size >= len(X_list)):
            c = 0
        yield X, y
        
        
# Data Augmentation method
def augmentation(imgs, masks): 
    for img, mask in zip(imgs, masks):
        img = cv2.imread("foetal-head-us/train/" + img)
        mask = cv2.imread("foetal-head-us/train/" + mask)
        train_img_aug.append(img)
        
        train_mask_aug.append(mask)
        img_lr = np.fliplr(img)
        mask_lr = np.fliplr(mask)
        img_up = np.flipud(img)
        mask_up = np.flipud(mask)
        img_lr_up = np.flipud(img_lr)
        mask_lr_up = np.flipud(mask_lr)
        img_up_lr = np.fliplr(img_up)
        mask_up_lr = np.fliplr(mask_up)
        train_img_aug.append(img_lr)
        train_mask_aug.append(mask_lr)
        train_img_aug.append(img_up)
        train_mask_aug.append(mask_up)
        train_img_aug.append(img_lr_up)
        train_mask_aug.append(mask_lr_up)
        train_img_aug.append(img_up_lr)
        train_mask_aug.append(mask_up_lr)
        
if __name__ == "__main__":
    # training data path
    path_train = config.PATH_TRAIN
    file_list_train = sorted(os.listdir(path_train))
    
    # test data path
    path_test = config.PATH_TEST
    file_list_test = sorted(os.listdir(path_test))
    
    # target image size
    IMG_HEIGHT = config.IMG_HEIGHT
    IMG_WIDTH = config.IMG_WIDTH
    
    smooth = config.smooth
    train_image = []
    train_mask = []
    for idx, item in enumerate(file_list_train):
        if idx % 2 == 0:
            train_image.append(item)
        else:
            train_mask.append(item)
    print("Number of US training images is {}".format(len(train_image)))
    print("Number of US training masks is {}".format(len(train_mask))) 
    
    ## Storing data
    X = []
    y = []
    for image, mask in zip(train_image, train_mask):
        X.append(np.array(Image.open(path_train+image)))
        y.append(np.array(Image.open(path_train+mask)))

    X = np.array(X)
    y = np.array(y)
    # data augmentation
    train_img_aug = []
    train_mask_aug = []
    augmentation(train_image, train_mask)
    
    #split training data
    X_train, X_val, y_train, y_val = train_test_split(train_img_aug, train_mask_aug, test_size = 0.2, random_state = 1)

    # set training parameters
    epochs = 50
    batch_size = 16
    steps_per_epoch = int(len(X_train) / batch_size)
    validation_steps = int(len(X_val) / batch_size)

    train_gen = Generator(X_train, y_train, batch_size = batch_size)
    val_gen = Generator(X_val, y_val, batch_size = batch_size)

    # initialize our model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    output_layer = build_model(inputs, 16, 0.5)

    # Define callbacks to save model with best val_dice_coef
    checkpointer = ModelCheckpoint(filepath = config.best_model_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
    model = Model(inputs=[inputs], outputs=[output_layer])
    model.compile(optimizer=Adam(lr = 3e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    results = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs = epochs,
                             validation_data = val_gen, validation_steps = validation_steps,callbacks=[checkpointer])
    
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(results.history['loss'], 'y', label='train loss')
    loss_ax.plot(results.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(results.history['dice_coef'], 'b', label='train dice coef')
    acc_ax.plot(results.history['val_dice_coef'], 'g', label='val dice coef')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.savefig("plot.png")

    plt.show()