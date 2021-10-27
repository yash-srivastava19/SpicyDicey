# !/usr/bin/python
import numpy 
import matplotlib.pyplot as plt
from glob import glob 
train_images,train_image_labels = [],[]
test_images ,test_image_labels = [],[]

train_list = [glob("Rotated/{}*.JPG".format(i)) for i in range(1,7)]

test_list = [glob("Resized/{}*.JPG".format(i)) for i in range(1,7)]

#Preparing Train Dataset
for names in train_list:
    for name in names:
        train_images.append(plt.imread(name))
        label = int(name.split("_")[0][-1])
        train_image_labels.append(label)


#Preparing Test Dataset
for names in test_list:
    for name in names:
        test_images.append(plt.imread(name))
        label = int(name.split("_")[0][-1])
        test_image_labels.append(label)

    

train_images = numpy.array(train_images)
train_image_labels = numpy.array(train_image_labels)


test_images = numpy.array(test_images)
test_image_labels = numpy.array(test_image_labels)


numpy.save("TrainDiceImages.npy",train_images)
numpy.save("TrainDiceLabels.npy",train_image_labels)

numpy.save("TestDiceImages.npy",test_images)
numpy.save("TestDiceLabels.npy",test_image_labels)
