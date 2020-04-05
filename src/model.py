# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 12:15:22 2020

@author: AshwinTR
"""

from tensorflow.python.keras import models 
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras import activations
from tensorflow.python.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.layers import *
import tensorflow.python.keras.backend as K
from PIL import Image
import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from data-compatibility import polyvore_dataset, DataGenerator
from utils import Config

#bug fix to shufffle data
class MyBugFix(tf.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()

#Learning rate scheduling
class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):  
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('New learning rate')
    
    def lr_schedule(epoch,lr):
        if epoch > 5:
            lr = lr/2
        return lr 

#model architecture
def get_my_model(width, height, depth):

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_shape = (height, width, depth)
    
        #split1 = Lambda(lambda x: x[:, :, :, :3])(data)
        #split2 = Lambda(lambda x: x[:, :, :, 3:])(data)
    
        input1 = Input(input_shape)
        input2 = Input(input_shape)
    
        model = Sequential()
        #model.add(Lambda(lambda x: x[:, :, :, :3]),input_shape=input_shape)
        model.add(Conv2D(32, (3,3), activation='relu',input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        #model.add(Dropout(0.5))

        part1 = model(input1)
        part2 = model(input2)

        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([part1, part2])
    
        output_layer = Dense( 1 , activation='sigmoid')(L1_distance)
        my_model = Model(inputs=[input1, input2], outputs=output_layer)

    return my_model

if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_valid, X_test, y_train, y_valid, y_test, n_classes = \
        dataset.X_train, dataset.X_valid, dataset.X_test, dataset.y_train, \
            dataset.y_valid, dataset.y_test, dataset.classes

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        valid_set = (X_valid[:100], y_valid[:100], transforms['test'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        valid_set = (X_valid, y_valid, transforms['test'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True,
              'learning_rate': Config['learning_rate']
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    valid_generator = DataGenerator(valid_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

    #compile model and fit data
    model = get_my_model(width=224, height=224, depth=3)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='pairwise_model.png', show_shapes=True, show_layer_names=True)

    results = model.fit(train_generator,validation_data=test_generator,epochs=Config['num_epochs'],
                        callbacks=[MyBugFix([train_generator.on_epoch_end]), LearningRateScheduler(lr_schedule)],
                        shuffle=True)

    model.save('/home/ubuntu/hw4/pairwise_model.hdf5')

    #get results
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    accuracy = results.history['accuracy']
    val_accuracy = results.history['val_accuracy']

    #plot learning curves
    epochs = np.arange(len(loss))
    plt.figure()
    plt.plot(epochs, loss, label='loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss for pairwise model')
    plt.legend()
    plt.savefig('/home/ubuntu/hw4/pairwise_model_loss.png', dpi=256)
    plt.close()
      
    plt.plot(epochs, accuracy, label='acc')
    plt.plot(epochs, val_accuracy, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for pairwise model')
    plt.legend()
    plt.savefig('/home/ubuntu/hw4/pairwise_model_acc.png', dpi=256)
    plt.close()


    
