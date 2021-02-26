import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from data import *
import tensorflow as tf
import keras.backend as K
import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#! Model import
from model_VGG3 import define_model
from model_VGG3 import define_model_adam
from model_VGG3 import define_model_lr
from model_VGG3 import define_model_dropout
from model_VGG3 import define_model_dropout_batchnorm

#! Hyper parameters
no_epochs = 100
batch_s = 64


experiment_name = "VGG3_"
data_folder = os.path.join(os.path.abspath(__file__),'results')

# plot diagnostic learning curves
def summarize_diagnostics(history, name=""):
    #! plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    #! plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    # save plot to file
    filename = experiment_name
    plt.savefig("results/" + filename + name + 'plot.pdf')
    plt.close()
    

#! create the confusion matrix
def create_confusion_matrix(model, testX ,testY, name):
    truth = testY.argmax(axis=1)
    predict =  model.predict(testX).argmax(axis=1)
    matrix = confusion_matrix(truth, predict)
    matrix = np.array2string(matrix, precision = 3, separator = ', ')
    matrix = matrix.replace(']',';')
    matrix = matrix.replace('[','')
    matrix = matrix.replace(';,',';')
    matrix = matrix.replace(';;',';')
    print(matrix, file=open("results/" + experiment_name + name + "confusion_matrix.txt", "w"))


#! run the test for evaluating the model based on depth with early stopping
def run_test_depth():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model_dropout_batchnorm()
    # fit model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    history = model.fit(trainX, trainY, epochs=no_epochs, batch_size=batch_s, validation_data=(testX, testY), verbose=1, callbacks=[es])
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    #compute the confusion matrix
    name = "dropout3_es_batchnorm_adam_data"
    create_confusion_matrix(model, testX, testY, name)
    # learning curves
    summarize_diagnostics(history, name)

#! Run test with dataugmentation 

def run_test_aug():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model_dropout_batchnorm()
    # fit model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience = 10)
    
    datagen = ImageDataGenerator(horizontal_flip=True)
    
    train = datagen.flow(trainX, trainY, batch_size=64)
        
    history = model.fit(x = train, batch_size= batch_s, epochs=no_epochs, validation_data=(testX, testY), verbose=1, callbacks=[es])
    
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    #compute the confusion matrix
    name = "dropout3_es_batchnorm_sgd_data"
    create_confusion_matrix(model, testX, testY, name)
    # learning curves
    summarize_diagnostics(history, name)

#! run the test for evaluating the model based on LR
def run_test_lr():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define models with different lr
    model1 = define_model_lr(0.001)
    model2 = define_model_lr(0.01)
    model3 = define_model_lr(0.1)
    # fit model
    history1 = model1.fit(trainX, trainY, epochs=no_epochs, batch_size=batch_s, validation_data=(testX, testY), verbose=1)
    history2 = model2.fit(trainX, trainY, epochs=no_epochs, batch_size=batch_s, validation_data=(testX, testY), verbose=1)
    history3 = model3.fit(trainX, trainY, epochs=no_epochs, batch_size=batch_s, validation_data=(testX, testY), verbose=1)
    # evaluate model
    _, acc = model1.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    _, acc = model2.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    _, acc = model3.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history1,"lr=0.001")
    summarize_diagnostics(history2,"lr=0.01")
    summarize_diagnostics(history3,"lr=0.1")


# entry point, run the test
run_test_aug()
