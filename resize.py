import os
from matplotlib.image import imread
import numpy as np
from PIL import Image


data_directory = './data_repository'
data_src="./data_repository/Veri776/"


def read_dataset(data_src):
    X = []
    y = []
    for directory in os.listdir(data_src):
        print(os.path.join(data_src, directory))
        try:
            for pic in os.listdir(os.path.join(data_src, directory)):
                img = Image.open(os.path.join(data_src, directory, pic))

                img=img.resize((28,28))

                a = np.asarray(img)
                print(img.size)
                #img = imread(os.path.join(data_src, directory, pic))
                #rows, cols = img.shape[:2]
                #print(rows,cols)
                X.append(np.squeeze(np.asarray(img)))
                y.append(directory)
        except Exception as e:
            print('Failed to read images from Directory: ', directory)
            print('Exception Message: ', e)
        break
    print('Dataset loaded successfully.')
    return X,y

X, y = read_dataset(data_src)
print(y)
