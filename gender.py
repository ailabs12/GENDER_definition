#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:36:38 2018
GENDER CLASSIFICATION BY FACES
@author: kdg
"""

import shutil
import os
import pandas as pd
import numpy as np
import time
""" Timing """
startTime = time.time()
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))

# destination function # male for 0, female for 1
Y = pd.read_csv('/home/kdg/kdg_projects/Gender/wiki.csv', sep=';')
gender = Y[' gender']
# finding Nan
gY = []
for g in gender:
    gY.append(float(g)) 
Y['gY'] = gY
data = Y.dropna()    # Dropping
data.shape          # Out[96]: (59685, 9)
gender = data.gY    # Output 
gender.sum()        # 47063 -> females, 12622 males :(
# equalizing gender quantity 
males = data[data.gY < 1]
females = data.drop(males.index)[:12622]        


# path to source
pic_path = data[' full_path']

# Каталог с набором данных
# data_dir = '/home/kdg/Gender/wiki_crop/'
# Каталог с данными для обучения
train_dir = '/home/kdg/kdg_projects/Gender/Face'
# Каталог с данными для проверки
val_dir = '/home/kdg/kdg_projects/Gender/Fval' 
# Каталог с данными для тестирования
test_dir = '/home/kdg/kdg_projects/Gender/Ftest'
# Часть набора данных для тестирования
test_data_portion = 0.15
# Часть набора данных для проверки
val_data_portion = 0.15
# Количество элементов данных в одном классе 12622
nb_images = 12622
# Функция создания каталога с двумя подкаталогами по названию классов: 
# male for 0, female for 1
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "male"))
    os.makedirs(os.path.join(dir_name, "female"))

# Создание структуры каталогов для обучающего, 
# проверочного и тестового набора данных
create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

# assembling all files to dst directory
dst = '/home/kdg/Gender/'    
for i, path in enumerate(pic_path):
    src = data_dir + path[1:]
    shutil.copy2(src, dst)  

# Расчет индексов наборов данных для обучения, прoверки и тестирования
start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
print(start_val_data_idx)       # 8835
print(start_test_data_idx)      # 10728

# Функция копирования изображений в заданный каталог. 
# Изображения (fe)males копируются в отдельные подкаталоги
s = 0
for i in males.index:
    shutil.copy2('/home/kdg/Gender/' + males[' full_path'].loc[i][4:], 
                    "/home/kdg/FAll/male")
    if s < start_val_data_idx:
        shutil.copy2('/home/kdg/Gender/' + males[' full_path'].loc[i][4:], 
                    train_dir + "/male")
    elif s < start_test_data_idx:
        shutil.copy2('/home/kdg/Gender/' + males[' full_path'].loc[i][4:], 
                    val_dir + "/male")   
    else: shutil.copy2('/home/kdg/Gender/' + males[' full_path'].loc[i][4:], 
                    test_dir + "/male")      
    s += 1
    
""" Для распознавания используется сверточная нейронная сеть 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 30
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 17670
# Количество изображений для проверки
nb_validation_samples = 3786
# Количество изображений для тестирования
nb_test_samples = 3788
"""
Создаем сверточную нейронную сеть
Архитектура сети:
Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., 
функция активации ReLU.
Слой подвыборки, выбор максимального значения из квадрата 2х2
Слой свертки, размер ядра 3х3, количество карт признаков - 32 шт., 
функция активации ReLU.
Слой подвыборки, выбор максимального значения из квадрата 2х2
Слой свертки, размер ядра 3х3, количество карт признаков - 64 шт., 
функция активации ReLU.
Слой подвыборки, выбор максимального значения из квадрата 2х2
Слой преобразования из двумерного в одномерное представление
Полносвязный слой, 64 нейрона, функция активации ReLU.
Слой Dropout.
Выходной слой, 1 нейрон, функция активации sigmoid
Слои с 1 по 6 используются для выделения важных признаков в изображении, 
а слои с 7 по 10 - для классификации.
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Компилируем нейронную сеть
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""
Создаем генератор изображений¶
Генератор изображений создается на основе класса ImageDataGenerator. 
Генератор делит значения всех пикселов изображения на 255 """
datagen = ImageDataGenerator(rescale=1. / 255)

# Генератор данных для обучения на основе изображений из каталога
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')    # Found 17670 images belonging to 2 classes

# Генератор данных для проверки на основе изображений из каталога
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')    # Found 3786 images belonging to 2 classes

# Генератор данных для тестирования на основе изображений из каталога

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')    # Found 3788 images belonging to 2 classes
"""
Обучаем модель с использованием генераторов
train_generator - генератор данных для обучения
validation_data - генератор данных для проверки """
model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = val_generator,
    validation_steps = nb_validation_samples // batch_size)
"""
Оцениваем качество работы сети с помощью генератора """
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
# Генерируем описание модели в json
model_json = model.to_json()
json_file = open('gender_model.json', 'w')
json_file.write(model_json)
json_file.close()
# Weights saving
model.save_weights('gender_model.h5')    

# Загрузка данных об архитектуре сети
json_file = open('gender_model.json', 'r')
loaded_model = json_file.read()
json_file.close()
# Создание модели на основе загруженных данных
new_model = model_from_json(loaded_model)
new_model.load_weigts('gender_model.h5')
new_model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Testing...
scores = new_model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

""" Распознавание пола на изображениях с помощью предварительно 
обученной нейронной сети VGG16 """
from tensorflow.python.keras.applications import VGG16
# OR
from keras.applications import VGG16
from keras.optimizers import Adam
# Размер мини-выборки
batch_size = 64

# Загружаем предварительно обученную нейронную сеть
vgg16_net = VGG16(weights='imagenet', 
                  include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
                  input_shape=(150, 150, 3))
# "Замораживаем" веса предварительно обученной нейронной сети VGG16
vgg16_net.trainable = False # Изображения мужчин/женщин уже там есть
vgg16_net.summary()
# Создаем составную нейронную сеть на основе VGG16
model = Sequential()
# Добавляем в модель сеть VGG16 вместо слоев свертки-пулинга;
# сами формируем лишь слои классификации
model.add(vgg16_net)    # выделение признаков
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# Компилируем составную нейронную сеть
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(lr=1e-5), # Низкая скорость обучения
              # если сделать больше - алгоритм не сойдется, т.к. он уже обучен
              # мы просто потеряем уже существующие веса
              metrics=['accuracy'])
""" GENERATORS """
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
"""
Обучаем модель с использованием генераторов """
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=5, # Поскольку сеть уже обучена
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

""" Тонкая настройка сети (fine tuning)
"Размораживаем" последний сверточный блок сети VGG16 """
vgg16_net.trainable = True
trainable = False
for layer in vgg16_net.layers:
    if layer.name == 'block5_conv1':
        trainable = True
    layer.trainable = trainable
model.summary()    

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5), 
              metrics=['accuracy'])

model.fit_generator(train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=3,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)    
# Аккуратность на тестовых данных: 83.40%

""" АНАЛИЗ ПРИЗНАКОВ, ИЗВЛЕЧЕННЫХ НЕЙРОСЕТЬЮ """
train_generator = datagen.flow_from_directory(train_dir,
    target_size=(224, 224), # standart for VGG16
    batch_size=10,
    class_mode=None,    # извлекаем только изображения, без ответов
    shuffle=False)      # тот же порядок, что на диске, без перемешивания
# CNN loading
vgg16 = VGG16(weights='imagenet', 
              include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
              input_shape=(224, 224, 3))
# извлечение признаков
features_train = vgg16.predict_generator(train_generator)
features_train.shape 
""" (17670, 7, 7, 512)
формат выходного значения сверточной части VGG:
17670 картинок, 512 карт признаков размером 7х7 """
# saving features    
np.save(open('features_train.npy','wb'), features_train)    
#
val_generator = datagen.flow_from_directory(val_dir,
    target_size=(224, 224), # standart for VGG16
    batch_size=10,
    class_mode=None,    # извлекаем только изображения, без ответов
    shuffle=False) 
startTime = time.time()
features_val = vgg16.predict_generator(val_generator)
features_val.shape  # (3788, 7, 7, 512)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
# Elapsed time: 150.428 sec
np.save(open('features_val.npy','wb'), features_val)  
#
test_generator = datagen.flow_from_directory(test_dir,
    target_size=(224, 224), # standart for VGG16
    batch_size=10,
    class_mode=None,    # извлекаем только изображения, без ответов
    shuffle=False) 
features_test = vgg16.predict_generator(test_generator)
features_test.shape  # (3788, 7, 7, 512)
np.save(open('features_test.npy','wb'), features_test)  

# Генерируем правильные ответы, 0 и 1 - метки классов
labels_train = np.array([0]*(nb_train_samples //2) + [1]*(nb_train_samples //2))
labels_val = np.array([0]*(nb_validation_samples //2) + [1]*(nb_validation_samples //2))
labels_test = np.array([0]*(nb_test_samples //2) + [1]*(nb_test_samples //2))
# Создаем простую сеть для классфикации
model = Sequential()
model.add(Flatten(input_shape=features_train.shape[1:]))
model.add(Dense(256, activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation ='sigmoid'))
#Компилируем нейронную сеть
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# RUN model
startTime = time.time()
model.fit(features_train, labels_train, epochs=15, batch_size=64,
          validation_data=(features_val, labels_val), verbose=2)  
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
