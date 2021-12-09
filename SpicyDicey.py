# !/usr/bin/python
#Create the datasets for this model by first running "Dataset.py" (adjust path if required)
#Use tensorboard for evaluating the model.
import os
import time
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
    LOGS_DIR = "/tmp/tb/tf_logs/Dice_MNIST/" + time.strftime('%d-%m-%Y_%H-%M-%S') 

config = Hyperparameters()

#Add your path to the numpy files here
xTrain,yTrain = numpy.load('TrainDiceImages.npy'), numpy.load('TrainDiceLabels.npy')
xTest,yTest = numpy.load('TestDiceImages.npy'), numpy.load('TestDiceLabels.npy')

xTrain = numpy.expand_dims(xTrain,-1)
xTest = numpy.expand_dims(xTest,-1)

yTrain = tf.keras.utils.to_categorical(yTrain,config.CLASSES)
yTest  = tf.keras.utils.to_categorical(yTest,config.CLASSES)

# It is recommended to double check these hyperparameters by tuning(preferably by using kerastuner)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape= config.INPUT_SHAPE,name = "InputLayer"),
    
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same',name = "Conv1),
    tf.keras.layers.MaxPooling2D(padding='same',name = "MaxPool1"),
                           
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding='same',name = "Conv2"),
    tf.keras.layers.MaxPooling2D(padding='same',name = "MaxPool2"),                   
    
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name = "Conv3"),
    tf.keras.layers.MaxPooling2D(padding='same',name = "MaxPool3"),                       
                           
    tf.keras.layers.Flatten(name = "FlatLayer1"),
    tf.keras.layers.Dropout(0.1,name = "Dropout1"),
                           
    tf.keras.layers.Dense(60,name = "Dense60")                       
    tf.keras.layers.Dense(config.CLASSES,activation='softmax',name = "Dense6")

])

#Debug: model.summary()
#Save : tf.keras.utils.plot_model('model.png',show_dtype = False, show_shapes = False, show_layer_names = True)
    
tBoardCallback = keras.callbacks.TensorBoard(config.LOGS_DIR,histogram_freq = 1, profile_batch = (500,520))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(xTrain,yTrain,batch_size=config.BATCH_SIZE ,epochs=config.EPOCHS,callbacks = [tBoardCallback])

loss,acc = model.evaluate(xTest,yTest)

#Debug: print("Loss:{}  Accuracy:{}".format(loss,acc))
#Save : model.save("DiceMNIST.h5")
