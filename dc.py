import numpy as np
import os
import re
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import optimizers

#STEP 1, Load Dataset Directory
dirname = os.path.join(os.getcwd(), sys.argv[1])
imgpath = dirname + os.sep 

images = [] #list of images loaded in memory
directories = [] #List of directorys in dataset, each directory should have images
                # from one class only and be named like the number of the class
dircount = []   #Images in each directory
prevRoot=''
cant=0

print("Reading images from... ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):

    #Directory founded
    if prevRoot !=root:

                prevRoot=root
                if len(filenames) > 0:
                    print("Directory founded:",root)
                    directories.append(root)
                cant=0

    for filename in filenames:
        
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename): #Check if there is an image with these formats
            #+1 to Total Images
            cant=cant+1 
            #Obtain Filepath of the image
            filepath = os.path.join(root, filename)
            #Load image in memory
            images.append( plt.imread(filepath) )
            b = "Reading..." + str(cant)
            print (b, end="\r")
    dircount.append(cant)

dircount = dircount[1:]
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

#STEP 2, Create Labels and Classes

labels = []

for i in range(len(dircount)):
    labels.extend( [i] * dircount[i] )

print("Lables created: ",len(labels))

classes_names=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    classes_names.append(name[len(name)-1])
    indice=indice+1

y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy

# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

#STEP 3, Train, Validation, Test and Preproccesing

train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

#Normalize images [0 - 1]
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

#STEP 4, Create CNN

INIT_LR = 1e-3
epochs = 10
batch_size = 64

sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(int(sys.argv[2]),int(sys.argv[2]),3)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2),padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(32, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5)) 
sport_model.add(Dense(nClasses, activation='softmax'))

sport_model.summary()

sport_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizers.RMSprop(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])

#STEP 5, Train CNN

sport_train_dropout = sport_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
sport_model.save("cd_mnist.h5py")

#STEP 6, Results

test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])