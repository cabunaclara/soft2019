import os
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt


#%matplotlib inline

import keras.backend as K
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from mlxtend.plotting import plot_confusion_matrix

seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)

input_path = "../../dataset/chest_xray/"
model_path="../models/myModel"
hist_path="../models/history.dat"

img_dims = 150
epochs = 10
batch_size = 32
val_batch_size=32


#Loading train, val and test data and doing augmentation
def process_data(image_dimensions, batch_size,val_batch_size):
    

    train_datagen = ImageDataGenerator(
                                        rescale= 1./255, 
                                        zoom_range= 0.3, 
                                        vertical_flip=True,
                                      )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
   
    train_gen = train_datagen.flow_from_directory(
        directory=input_path+'train', 
        target_size=(image_dimensions, image_dimensions), 
        batch_size=batch_size, 
        class_mode='binary', 
        shuffle=True)
    
    
    test_gen = test_datagen.flow_from_directory(
        directory=input_path+'test', 
        target_size=(image_dimensions,image_dimensions), 
        batch_size=batch_size, 
        class_mode='binary', 
        shuffle=True)
    
    
    val_gen = test_datagen.flow_from_directory(
        directory=input_path+'val', 
        target_size=(image_dimensions,image_dimensions), 
        batch_size=val_batch_size, 
        class_mode='binary', 
        shuffle=True)
    

    # list that are going to contain test images data and the corresponding labels
    test_data = []
    test_labels = []
    
    
    # list that are going to contain validation images data and the corresponding labels
    val_data = []
    val_labels = []


    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path+'test'+cond+img)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/NORMAL/':
                label = 0
            elif cond=='/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
       
    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + 'val' + cond)):
            img = plt.imread(input_path+'val'+cond+img)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/NORMAL/':
                label = 0
            elif cond=='/PNEUMONIA/':
                label = 1
            val_data.append(img)
            val_labels.append(label)


    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    
    return train_gen, test_gen, test_data, test_labels, val_gen, val_data, val_labels

#Loading trained model
def load_model():
    model = keras.models.load_model(model_path)
    return model

#Saving model
def save_model(model):
    model.save(model_path)
   
#Loading history
def load_history_from_file():
    return pickle.load(open(hist_path, mode="rb"))

#Saving history
def save_history(history):
    pickle.dump(history, open(hist_path, mode="wb"))

#Creating convolutional and fully conected layers and creating model
def build_model(img_dims):
    inputs = Input(shape=(img_dims, img_dims, 3))
    
    # First conv block
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    # Second conv block
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    # Third conv block
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    # Fourth conv block
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)
    
    # Fifth conv block
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)
    
    # FC layer
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.7)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    
    # Output layer
    output = Dense(units=1, activation='sigmoid')(x)
    
    # Creating model and compiling
    model = Model(inputs=inputs, outputs=output)
    #model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

#Model training
def train(model,train_gen,val_gen):
    
    # Callbacks
    checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
    #early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

    hist = model.fit_generator(
            train_gen, steps_per_epoch=train_gen.samples // batch_size, 
            epochs=epochs, validation_data=val_gen, 
            validation_steps=val_gen.samples // batch_size, callbacks=[checkpoint, lr_reduce])

    save_model(model);
    save_history(hist)
    return hist

#Prediction and evaluation
def predict_and_evaluate(model, test_data, test_labels, hist):
    preds = model.predict(test_data)

    acc = accuracy_score(test_labels, np.round(preds))*100
    cm = confusion_matrix(test_labels, np.round(preds))
    tn, fp, fn, tp = cm.ravel()
    
    print('CONFUSION MATRIX ------------------')
    # print(cm)
    # print ("true negative:")
    # print(tn)
    # print ("true positive:")
    # print(tp)
    # print("false negative:")
    # print(fn)
    # print ("false positive:")
    # print(fp)
    plot_cm(cm)
  
    
    print('\nTEST METRICS ----------------------')
    precision = tp/(tp+fp)*100
    recall = tp/(tp+fn)*100
    print('Accuracy: {}%'.format(acc))
    print('Precision: {}%'.format(precision))
    print('Recall: {}%'.format(recall))
    print('F1-score: {}'.format(2*precision*recall/(precision+recall)))
    
    print('\nTRAIN METRIC ----------------------')
    print('Train acc: {}'.format(np.round((hist.history['accuracy'][-1])*100, 2)))



#Plot acuracy and loss
def create_plots_acc_loss(hist):
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(hist.history[met])
        ax[i].plot(hist.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train', 'val'])

#Plot confusion matrix
def plot_cm(cm): 
    
    class_names=["Normal","Pneumonia"]
   
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True,
                                show_normed=True,colorbar=True,
                                class_names=class_names)
    plt.show()

#Plot images 
def plot_image_sample():
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax = ax.ravel()
    plt.tight_layout()
    
    for i, _set in enumerate(['train', 'val', 'test']):
        set_path = input_path+_set
        ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]), cmap='gray')
        ax[i].set_title('Set: {}, Condition: Normal'.format(_set))
        ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]), cmap='gray')
        ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))


    for _set in ['train', 'val', 'test']:
        n_normal = len(os.listdir(input_path + _set + '/NORMAL'))
        n_infect = len(os.listdir(input_path + _set + '/PNEUMONIA'))
        print('Set: {}, normal images: {}, pneumonia images: {}'.format(_set, n_normal, n_infect))

    
train_gen, test_gen, test_data, test_labels, val_gen, val_data, val_labels = process_data(img_dims, batch_size, val_batch_size)

model = build_model(img_dims)
hist = train(model,train_gen,test_gen)

#remove comment for loading model 
#model = load_model()

#remove comment for loading history 
#hist=load_history_from_file()

create_plots_acc_loss(hist)
predict_and_evaluate(model, test_data, test_labels, hist)









 