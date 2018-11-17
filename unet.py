from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import cv2

TRAINING_SIZE = 522
VALIDATION_SIZE = 100

DIMEN_X = 100
DIMEN_Y = 100
PREPROCESS = False



def main():
    if PREPROCESS:
        preprocess()
    model = modelo()
    X,Y,Xval,Yval = gen_dataset()
    model.fit(X,Y, batch_size = 2, validation_data=(Xval,Yval), epochs = 300)
    layer = model.layers[-1]
    print("layer: ", layer)
    print("layer type: ", type(layer))
    print("layer_output: ", layer.output)
    print("layer_output_type: ", type(layer.output))
    print("layer_output.get_value: ", tf.Print(layer.output, [layer.output[0,0]]))
    print("layer_output.get_value type: ", type(tf.Print(layer.output, [layer.output[0,0]])))


def gen_dataset():
    X = np.empty((TRAINING_SIZE,DIMEN_X,DIMEN_Y,3))
    Xval = np.empty((VALIDATION_SIZE,DIMEN_X,DIMEN_Y,3))
    Y = np.empty((TRAINING_SIZE,DIMEN_X,DIMEN_Y,1))
    Yval = np.empty((VALIDATION_SIZE,DIMEN_X,DIMEN_Y,1))
    os.chdir("Data/training/image")
    files = os.listdir()
    files.sort()
    for i in range(len(files)):
        file = files[i]
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (DIMEN_X,DIMEN_Y))
        X[i] = resized_image
    os.chdir("../gtf")
    files = os.listdir()
    files.sort()
    for i in range(len(files)):
        file = files[i]
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (DIMEN_X,DIMEN_Y))
        resized_image = np.expand_dims(resized_image, axis=2)
        print("shape: ", image.shape)
        Y[i] = (resized_image/255)

    os.chdir("../../val/image")
    files = os.listdir()
    files.sort()
    print("dir: ", os.listdir())
    print("files: ", os.listdir())
    for i in range(len(files)):
        file = files[i]
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (DIMEN_X,DIMEN_Y))
        Xval[i] = (resized_image/255)
    
    os.chdir("../gtf")
    files = os.listdir()
    files.sort()
    print("dir: ", os.listdir())
    print("files: ", os.listdir())
    for i in range(len(files)):
        file = files[i]
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (DIMEN_X,DIMEN_Y))
        resized_image = np.expand_dims(resized_image, axis=2)
        Yval[i] = (resized_image/255)

    return (X,Y, Xval, Yval)








def preprocess():
    os.chdir("Data/")
    print(os.listdir())
    os.chdir("val/")
    print(os.listdir())
    os.chdir("gt/")
    files = os.listdir()
    files.sort()
    for file in files:
        image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        print(type(image))
        print(image.shape)
        print(type(file))
        timage = transform(file, image)
                




def transform(file, image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j] != 0):
                image[i][j] = 255
    os.chdir("../gtf/")
    cv2.imwrite(file,image)
    os.chdir("../gt/")


def modelo():
    input_size = (DIMEN_X,DIMEN_Y,3)
    concat_axis = 3

    inputs = Input(input_size)

    # Primer nivel
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name="c1")(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name="c2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Segundo nivel
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name="c3")(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name="c4")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Tercer nivel
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name="c5")(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name="c6")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Cuarto nivel
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name="c7")(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name="c8")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Quinto nivel
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name="c9")(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name="c10")(conv5)

    # Primero subida
    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5) # Tamaño de redimensionamiento
    # Redimensionamiento
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4) 
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name="c11")(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name="c12")(conv6)

    # Segundo subida
    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6) # Redimensionamiento
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis) 
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name="c13")(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name="c14")(conv7)

    # Tercero subida
    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name="c15")(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name="c16")(conv8)

    # Cuarto subida
    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name="c17")(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name="c18")(conv9)

    # Quinta subida
    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = Conv2D(1, (1, 1), name="c19")(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = Conv2D(1, (1, 1), name="c20")(conv9)
    print(conv10.get_shape())



    model = Model(inputs=inputs, outputs=conv10)
    
    # Compilamos el modelo (optimizador y medida de perdida)
    model.compile(optimizer = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss = 'binary_crossentropy', metrics = ['accuracy'])

    
    print("termino!")
    return model

def get_crop_shape(current, target):
    # alto
    ch = (current.get_shape()[1] - target.get_shape()[1]).value
    assert(ch >= 0)
    # impar
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else: 
        ch1, ch2 = int(ch/2), int(ch/2)
    
    # ancho
    cw = (current.get_shape()[2] - target.get_shape()[2]).value
    assert (cw >= 0)
    # impar
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) +1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    
    return (ch1, ch2), (cw1, cw2)



def another():
    os.chdir("Data/training/gtf")
    files = os.listdir()
    files.sort()
    file = files[0]
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (DIMEN_X,DIMEN_Y))
    resized_image = np.expand_dims(resized_image, axis=2)
    print("shape: ", image.shape)
    print("imagen: ", image)

    

if __name__ == "__main__":
    main()