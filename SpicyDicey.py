# !/usr/bin/python
#Create the datasets for this model by first running "Dataset.py" (adjust path if required)
#Use tensorboard for evaluating the model.
import os
import numpy
import tensorflow as tf
from dataclasses import dataclass
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

@dataclass
class Hyperparameters:
    CLASSES = 7
    INPUT_SHAPE = (28,28,1)
    BATCH_SIZE = 5
    EPOCHS = 100
    LOGS_DIR = "/tmp/tb/tf_logs/"

config = Hyperparameters()

#Add your path to the numpy files here
xTrain,yTrain = numpy.load("TrainDiceImages.npy"),numpy.load("TrainDiceLabels.npy")

xTest,yTest = numpy.load("TestDiceImages.npy"),numpy.load("TestDiceLabels.npy")


xTrain = xTrain.astype('float32')/255
xTrain = numpy.expand_dims(xTrain,-1)

xTest = xTest.astype('float32')/255
xTest = numpy.expand_dims(xTest,-1)


yTrain = tf.keras.utils.to_categorical(yTrain,config.CLASSES)
yTest  = tf.keras.utils.to_categorical(yTest,config.CLASSES)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape= config.INPUT_SHAPE),
    tf.keras.layers.Conv2D(16,kernel_size=(3,3),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Conv2D(32,kernel_size=(1,1),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(config.CLASSES,activation='softmax')

])

#Debug: model.summary()
tBoardCallback = keras.callbacks.TensorBoard(config.LOGS_DIR,histogram_freq = 1, profile_batch = (500,520))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(xTrain,yTrain,batch_size=config.BATCH_SIZE ,epochs=config.EPOCHS,callbacks = [tBoardCallback])

loss,acc = model.evaluate(xTest,yTest)

#Debug: print("Loss:{}  Accuracy:{}".format(loss,acc))
