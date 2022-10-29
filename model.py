import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.io import imread, imshow
from tqdm import tqdm

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = r'C:\Users\Allone_chy\Desktop\Unet\raw\\'
MASK_PATH = r"C:\Users\Allone_chy\Desktop\Unet\masks\\"

list(os.walk(TRAIN_PATH))

train_ids = next(os.walk(TRAIN_PATH))

mask_ids = next(os.walk(MASK_PATH))
# mask_ids[2]

# len(train_ids[2])

X_train = np.zeros((len(train_ids[2]),IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

for n, id_ in tqdm(enumerate(train_ids[2]),total = len(train_ids[2])):
    path = r'C:\Users\Allone_chy\Desktop\Unet\raw\\'
    img = imread(path+id_)
    X_train[n] = img

# imshow(X_train[0])
# plt.show()

Y_train = np.zeros((len(train_ids[2]),IMG_HEIGHT, IMG_WIDTH, 1), dtype = bool)

for n, mask_id_ in tqdm(enumerate(mask_ids[2]),total = len(mask_ids[2])):
    path = r'C:\Users\Allone_chy\Desktop\Unet\masks\\'
    mask_ = imread(path + mask_id_)
    mask_ = np.expand_dims(mask_, axis = -1)
    Y_train[n] = mask_
    #print(n, mask_id_)

# imshow(np.squeeze(Y_train[0]))
# plt.show()

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x:x /255)(inputs)

c1 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c5)

u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u6)
c6 = tf.keras.layers.Dropout(0.1)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u7)
c7 = tf.keras.layers.Dropout(0.1)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides = (2,2), padding = 'same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
c9 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c9)
outputs = tf.keras.layers.Conv2D(1,(1,1), activation = 'sigmoid')(c9)

model = tf.keras.Model(inputs =[inputs], outputs = [outputs])
model.compile(optimizer ='adam', loss ='binary_crossentropy', metrics = ['accuracy'])
# model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_allen_mouse_brain_ISH.h5', verbose = 1, save_best_only = True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'logs')]

trainsize = 70

X_train_rand = np.zeros((trainsize,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train_rand = np.zeros((trainsize,IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.uint8)

randindex = random.sample(range(180), trainsize)

for i in range(trainsize):
    n = randindex[i]
    X_train_rand[i] = X_train[n]
    Y_train_rand[i] = Y_train[n]

results = model.fit(X_train_rand, Y_train_rand, validation_split = 0.1, batch_size = 16, epochs = 25, callbacks = callbacks)



# preds_val = model.predict(X_train[0:4],verbose = 1)
# for i in range(4):
#     imshow((preds_val[i]>0.05).astype(np.uint8))
#     plt.show()
# # preds_val= model.predict(X_train[41:50], verbose = 1)





# TEST_PATH = r'C:\Users\Allone_chy\Desktop\Unet\validation\\'

# list(os.walk(TEST_PATH))

# test_ids = next(os.walk(TEST_PATH))

# test_list = np.zeros((len(test_ids[2]),IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

# for n, id_ in tqdm(enumerate(test_ids[2]),total = len(test_ids[2])):
#     path = r'C:\Users\Allone_chy\Desktop\Unet\validation\\'
#     img = imread(path+id_)
#     test_list[n] = img


img = cv.imread("./img2.jpg")

height = img.shape[0]
width = img.shape[1]

height_seg = height // IMG_HEIGHT
width_seg = width // IMG_WIDTH
num_seg = height_seg * width_seg

valid_set = np.zeros((num_seg, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)

for h in range(height_seg):
    for w in range(width_seg):
        imgtemp = img[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH]
        index = h * width_seg + w
        valid_set[index] = imgtemp

preds_train = model.predict(valid_set, verbose = 1)


for n in range(num_seg):
    imshow((preds_train[n]>0.01).astype(np.uint8))
    plt.show()

    imshow(X_train[n])
    plt.show()
