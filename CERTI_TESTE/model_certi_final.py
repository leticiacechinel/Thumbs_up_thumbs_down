import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras import models, layers, optimizers
from keras.models import Model
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random


############ DATA AUGMENTATION, OBTENDO MAIS DADOS E GENERALIZANDO DATASET ####################


datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

path = 'thumbs/'
for folder in os.listdir(path):
    path_ = path + folder + '/'
    # print("FOLDER ",folder)
    for file in os.listdir(path_):
        # print(file)
        if file.endswith('.jpeg') and not "_gen" in file:
            
            img = load_img(path_+file) 
            x = img_to_array(img)  
            x = x.reshape((1,) + x.shape)

            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='thumbs/'+folder, save_prefix=str(folder)+'_gen', save_format='jpeg'):
                i += 1
                if i > 5:
                    break  # Recurso de parada do gerador


########### LEITURA DE ARQUIVOS, TRANSFORMACAO 1 CANAL, TRANSFORMACAO PARA PONTO FLUTUANTE E NORMALIZACAO ###########


path = 'thumbs/up'
images = []
labels = []
images_color = []

for file in os.listdir(path):
    img = cv2.imread(path+'/'+file)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    resized = cv2.resize(gray, (100, 100))
    resized = np.append(resized,1)
    images.append(resized)
    
path2 = path = 'thumbs/down'
for file in os.listdir(path2):
    img = cv2.imread(path2+'/'+file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    resized = cv2.resize(gray, (100, 100))
    resized = np.append(resized,0)
    images.append(resized)


random.shuffle(images)

labels = np.asarray([imagem[-1] for imagem in images])
imagens = np.asarray([imagem[0:-1].reshape(100,100,1) for imagem in images])


imagens = imagens.astype('float32')
imagens  =  (imagens) / 255


########### SEPARACAO TREINO - TESTE E TREINAMENTO DO MODELO ###########

index_split = round(0.8*len(imagens))

train_images = imagens[0:index_split]
test_images = imagens[index_split::]
train_labels = labels[0:index_split]
test_labels = labels[index_split::]

# with tf.device('/device:XLA_GPU:0'):
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=150,validation_data=(test_images, test_labels))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('./modelo_teste')