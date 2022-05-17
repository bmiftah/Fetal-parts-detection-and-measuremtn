PATH_TRAIN = "train/"
PATH_TEST = "Images_for_Testing"
# PATH_TEST ="abdomen_data"
# RESULT = "abdomen_data_mask"
#/content/drive/MyDrive/foetal-head-segmentation/foetal-head-segmentation
#/content/drive/MyDrive/foetal-head-segmentation/foetal-head-segmentation/test_set4
# set target image size
IMG_HEIGHT = 224
IMG_WIDTH = 224

epochs = 50
batch_size = 16

#division smooth
smooth = 1.

best_model_path = "best_model_224_res.h5"

### 01best .....
# -----> currently installed keras version 2.2.4-tf
# tensorflow version 2.0.0
#  opencv version 4.5.3
# python version 3.6.15 (default, Dec  3 2021, 18:25:24) [MSC v.1916 64 bit (AMD64)]
# numpy version 1.19.5
# ----> Model's keras version  2.8.0

# best_model ...
# ----> currently installed keras version 2.2.4-tf
# tensorflow version 2.0.0
#  opencv version 4.5.3
# python version 3.6.15 (default, Dec  3 2021, 18:25:24) [MSC v.1916 64 bit (AMD64)]
# numpy version 1.19.5
# ---> Model's keras version  2.8.0

# ---> model trained on colab
# currently installed keras version 2.8.0 - colab
# tensorflow version 2.8.0
# keras version 2.8.0
#  opencv version 4.1.2
# python version 3.7.13 (default, Mar 16 2022, 17:37:17)
# [GCC 7.5.0]
# numpy version 1.21.5
# Model's keras version  2.8.0
