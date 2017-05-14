# -*- coding: utf-8 -*-
"""
Behavioural Cloning Project

Author: Michael Matthews
"""

import os
import csv
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D
from keras.layers import Lambda, Cropping2D, Dropout
from keras import optimizers

# Build the list of CSV file folders.
DRIVING_CSV_FOLDER = "/home/michael/CarND-Behavioral-Cloning-P3"
DRIVING_CSV_FILES = [f for f in os.listdir(DRIVING_CSV_FOLDER)
                             if os.path.isdir(os.path.join(DRIVING_CSV_FOLDER,f)) and
                                f.startswith('track')]

# Display the histogram in output PDF?
SHOW_HISTOGRAM = True

# Correction factor to turn left/right images towards a centre image,
SIDE_CORRECTION = 0.1

# Split for Training and Validation data.
VALIDATION_SPLIT = 0.2

# Activation function to use ('relu' or 'elu')
ACTIVATION = 'relu'

def create_data():
    global SHOW_HISTOGRAM

    # Set the seed for random numbers.  Keeping this constant helps for model rebuilds.
    random.seed(42)
    
    images = []
    measurements = []
    for driving_csv in DRIVING_CSV_FILES:
        file = os.path.join(DRIVING_CSV_FOLDER, driving_csv, "driving_log.csv")
        print("Reading CSV file:", file)
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                
                # Add the centre camera view.
                image = line[0]
                measurement = float(line[3])
                
                if abs(measurement) < 0.075:
                    if random.random()>0.25:
                        continue
    
                if os.path.isfile(image):
                    images.append(cv2.imread(image)[:,:,::-1]) # Convert BRG to RGB

                    measurements.append(measurement)
                    
                    # Add a flipped version of the image to increase training data,
                    images.append(np.fliplr(images[-1]))
                    measurements.append(-measurements[-1])
    
                for image, offset in zip([line[1], line[2]],
                                         [SIDE_CORRECTION, -SIDE_CORRECTION]):
                    # Add the image and measurement.
                    images.append(cv2.imread(image)[:,:,::-1]) # Convert BRG to RGB
                    measurements.append(float(line[3])+offset)
                    
                    # Add a flipped version of the image to increase training data,
                    images.append(np.fliplr(images[-1]))
                    measurements.append(-measurements[-1])

    X_train = np.array(images)
    y_train = np.array(measurements)

    if SHOW_HISTOGRAM:
        fig, ax = plt.subplots()
        ax.hist(y_train, bins=21)
        ax.set_title('Histogram of driving angle measurements')
        ax.set_ylabel('count')
        ax.set_xlabel('angle')
        plotpdf.savefig(fig)
        plt.close(fig)
        SHOW_HISTOGRAM = False

    return X_train, y_train

def train_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1))
    
    if NETWORK == "lenet": # Starts with basic steering but does not reach the bridge.
        model.add(Convolution2D(6,(5,5),activation='relu'))
        model.add(MaxPooling2D())
        model.add(Convolution2D(6,(5,5),activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))
    elif NETWORK == "nvidia":
        model.add(Convolution2D(24,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Convolution2D(36,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Convolution2D(48,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Convolution2D(64,(3,3),activation=ACTIVATION))
        model.add(Convolution2D(64,(3,3),activation=ACTIVATION))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
    elif NETWORK == "custom1":
        model.add(Convolution2D(24,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Convolution2D(36,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Convolution2D(48,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Convolution2D(64,(3,3),activation=ACTIVATION))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Convolution2D(64,(3,3),activation=ACTIVATION))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
    elif NETWORK == "custom2":
        model.add(Convolution2D(24,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Convolution2D(36,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Convolution2D(48,(5,5),strides=(2,2),activation=ACTIVATION))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Convolution2D(64,(3,3),activation=ACTIVATION))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Convolution2D(64,(3,3),activation=ACTIVATION))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Dense(50))
        model.add(Dropout(DROPOUT)) # Added ...
        model.add(Dense(10))
        model.add(Dense(1))
    else:
        raise NotImplementedError("Unknown NETWORK of type '" + str(NETWORK) + "'.")
    
    adam = optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    history_object = model.fit(X_train, y_train,
                               validation_split=VALIDATION_SPLIT,
                               shuffle=True,
                               batch_size=BATCH_SIZE,
                               epochs=EPOCHS)
    
    model.save("/home/michael/CarND-Behavioral-Cloning-P3/model.h5")
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    fig, ax = plt.subplots()
    ax.plot(history_object.history['loss'])
    ax.plot(history_object.history['val_loss'])
    ax.set_title('Model:' + NETWORK + ', Batch Size:' + str(BATCH_SIZE) +
                 ', Learn Rate:' + str(LEARNING_RATE) +
                 ('' if DROPOUT is None else ', Dropout:' + str(DROPOUT)))
    ax.set_ylabel('mean squared error loss')
    ax.set_xlabel('epoch')
    ax.legend(['training set', 'validation set'], loc='upper right')
    plotpdf.savefig(fig)
    plt.close(fig)

# Read the source CSV
EPOCHS = 10

if 0:
    plotpdf = PdfPages('model.pdf')
    
    X_train, y_train = create_data()

    for LEARNING_RATE in [0.001, 0.0001]:
        for BATCH_SIZE in [64, 128, 256, 512]:
            for NETWORK, DROPOUT in [["nvidia", None], ["custom1", 0.1],
                                     ["custom1", 0.2], ["custom2", 0.1],
                                     ["custom2", 0.2]]:
                print("Training NETWORK:", NETWORK, "DROPOUT:", DROPOUT,
                      "BATCH_SIZE:", BATCH_SIZE, "LEARNING_RATE:", LEARNING_RATE)
                train_model()
    
    plotpdf.close()

if 1:
    plotpdf = PdfPages('model_final.pdf')
    
    X_train, y_train = create_data()
    
    NETWORK = "custom2"
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 512
    DROPOUT = 0.2
    print("Training NETWORK:", NETWORK, "DROPOUT:", DROPOUT,
          "BATCH_SIZE:", BATCH_SIZE, "LEARNING_RATE:", LEARNING_RATE)
    train_model()
    
    plotpdf.close()
