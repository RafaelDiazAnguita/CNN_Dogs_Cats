import os
import re
import sys
from PIL import Image

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
            image = Image.open(filepath)
            image = image.resize((int(sys.argv[2]),int(sys.argv[2])))
            image.save(filepath)
            b = "Reading..." + str(cant)
            print (b, end="\r")
    dircount.append(cant)

dircount = dircount[1:]
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

