import config
import Post_processing_for_Ellipse_fitting
import tensorflow.keras
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from metrics import dice_coef, dice_coef_loss
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import tensorflow as tf
import random as rng
rng.seed(56)
color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
#import keras
# import keras
# print(keras.__version__)'
from PIL import Image
#from google.colab.patches import cv2_imshow
#smooth = config.smooth
path_test = config.PATH_TEST
IMG_HEIGHT = config.IMG_HEIGHT
IMG_WIDTH = config.IMG_WIDTH
f = h5py.File('best_model_224_res.h5', 'r') # loading pre-trained model
path = "Images_for_Testing/"
Input_image_list = os.listdir(path_test)
#print(" List of test data : ", test_list)

## The below print statements can be removed - they just print currently used libraries and frame-work version
print("currently installed keras version",tensorflow.keras. __version__)
print("tensorflow version", tf.__version__)
#print("keras version",keras. __version__)
print(" opencv version",cv2. __version__)
print("python version",sys.version)
print ("numpy impoversion",np.version.version)

print("Model's keras version ",f.attrs.get('keras_version'))

model = load_model(config.best_model_path,custom_objects={"dice_coef_loss":dice_coef_loss, "dice_coef":dice_coef})


X_test = np.empty((len(Input_image_list), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
for i, item in enumerate(Input_image_list):
    print("i ",i," item ",str(path_test)+str("/")+str(item))
    image = cv2.imread(str(path_test)+ "/" + str(item), 0)
    print(" image shape ",image.shape)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
    X_test[i] = image
  
X_test = X_test[:,:,:,np.newaxis] / 255

y_pred = model.predict(X_test)
#print("X Test ",len(X_test)," Y_Pred ",len(y_pred))
image_count = 0

print(" displaying / saving data ... ")
for test_img, prediction in zip(X_test[0:11],y_pred[0:11]):
    # print(" >>>>>>>  ")
    # print(" test shape ",test.shape)
    # print(" pred shape ",pred.shape)

    test = test_img.reshape((IMG_HEIGHT,IMG_WIDTH))
    filename,file_extension = os.path.splitext(str(Input_image_list[image_count]))

    prediction = prediction.reshape((IMG_HEIGHT,IMG_WIDTH))
    prediction = prediction > 0.5
    prediction = np.ma.masked_where(prediction == 0, prediction)

    predicted_mask = ((np.array(prediction))*255.0).astype(np.uint8)
    Input_image = ((np.array(test))*255.0).astype(np.uint8)


    cv2.imwrite('Mask_prediciton_Result/Mask_{}.png'.format(filename),predicted_mask)  # saving mask
    cv2.imwrite('Mask_only/Mask_{}.png'.format(filename), predicted_mask)              # saving mask for Ellipse fitting
    cv2.imwrite('Mask_prediciton_Result/Image_{}.png'.format(filename),Input_image)  # saving image

    im3 = cv2.addWeighted(Input_image, 1.0, predicted_mask, 0.4, 1) # dtype = cv2.CV_32F  # Image and Mask over_layed
    cv2.imwrite('Mask_prediciton_Result/Image_Mask_{}.png'.format(filename),im3)  # Image mereged with mask

    color = (255, 255, 255)

    source_window1 = 'EMP  '
    source_window2 = 'EMP  '
    source_window3 = 'EMP  '

    cv2.namedWindow(source_window1)

    cv2.imshow(source_window1, Input_image)
    cv2.waitKey(1000)


    cv2.imshow(source_window2,predicted_mask)
    cv2.waitKey(1000)


    cv2.moveWindow("image 2", 0, 2)
    cv2.imshow(source_window3,im3)
    cv2.waitKey(6000)

    Post_processing_for_Ellipse_fitting.main(Input_image,predicted_mask,image_count)
    image_count += 1
