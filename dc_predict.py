import numpy as np
import os
import re
import sys
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('cd_mnist.h5py')

images=[]
filenames = []
for i in range(8):
    filenames.append(str(i+1)+".jpg")


for filepath in filenames:
    image = plt.imread(filepath)
    images.append(image)

X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
test_X = X.astype('float32')
test_X = test_X / 255.

predicted_classes = model.predict(test_X)

print("PREDICTION: " ,predicted_classes)