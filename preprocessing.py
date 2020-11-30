import os
from matplotlib.image import imread
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

def main_train(src):
    
    images_train = np.array([])
    images_val = np.array([])
    labels_train = np.array([])
    labels_val = np.array([])
    unique_train_label = np.array([])
    map_train_indices = dict()
    
    images_train, images_val, labels_train,labels_val = preprocessing1(0.9, src)
    unique_train_label = np.unique(labels_train)
    map_train_indices = {label: np.flatnonzero(labels_train == label) for label in unique_train_label}
    print('Preprocessing Done. Summary:')
    print("Images train :", images_train.shape)
    print("Labels train :", labels_train.shape)
    print("Images val  :", images_val.shape)
    print("Labels val  :", labels_val.shape)
    print("Unique label :", unique_train_label)
    return images_train, labels_train, images_val, labels_val, unique_train_label, map_train_indices
    
       
def main_test(src):
    images_test = np.array([])
    labels_test = np.array([])
    unique_test_label = np.array([])
    map_test_indices = dict()
    
    images_test, labels_test = preprocessing2(src)
    unique_test_label = np.unique(labels_test)
    map_test_indices = {label: np.flatnonzero(labels_test == label) for label in unique_test_label}
    print('Preprocessing Done. Summary:')
    print("Images test  :", images_test.shape)
    print("Labels test  :", labels_test.shape)
    print("Unique label :", unique_test_label)
    return images_test, labels_test, unique_test_label, map_test_indices
     
    

def read_dataset(src):
        X = []
        y = []
        data_src = src
        for directory in os.listdir(data_src):
            print(directory)
            path1 = data_src+directory+'/'
            try:        
                for pic in os.listdir(path1):
                    path = path1+pic
                    #print(path)
                    #img = imread(os.path.join(self.data_src, directory, pic))
                    img = cv2.imread(path)

                    #img=cv2.resize(img, (122,122))
                    img=cv2.resize(img, (32, 32))                    


                    X.append(np.squeeze(np.asarray(img)))
                    y.append(directory)
            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return X,y

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x

def preprocessing1(train_test_ratio, src):
        X, y = read_dataset(src)
        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y])
        X = [normalize(x) for x in X]                                  # normalize images

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = []
        y_shuffled = []
        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])
        size_of_dataset = len(x_shuffled)
        x_shuffled=np.array(x_shuffled, dtype=np.float32)
        n_train = int(np.ceil(size_of_dataset * train_test_ratio))
        return np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(y_shuffled[0:n_train]), np.asarray(y_shuffled[n_train + 1:size_of_dataset])
    
def preprocessing2(src):
        X, y = read_dataset(src)
        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y])
        X = [normalize(x) for x in X]                                  # normalize images

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = []
        y_shuffled = []
        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])
            
        size_of_dataset = len(x_shuffled)
        x_shuffled=np.array(x_shuffled, dtype=np.float32)
        n_test = int(np.ceil(size_of_dataset))
        
        return np.asarray(x_shuffled[0:n_test]), np.asarray(y_shuffled[0:n_test])

def get_triplets(unique_train_label, map_train_indices):
        label_l, label_r = np.random.choice(unique_train_label, 2, replace=False)
        a, p = np.random.choice(map_train_indices[label_l],2, replace=False)
        n = np.random.choice(map_train_indices[label_r])
        #a = tf.convert_to_tensor(a)
        #p = tf.convert_to_tensor(p)
        #n = tf.convert_to_tensor(n)
        return a, p, n

def get_triplets_batch(n, images_train, unique_train_label, map_train_indices):
        idxs_a, idxs_p, idxs_n = [], [], []
        for _ in range(n):
            a, p, n = get_triplets(unique_train_label,map_train_indices)
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)
            a = tf.convert_to_tensor(images_train[idxs_a,:], dtype=tf.float32)
            p = tf.convert_to_tensor(images_train[idxs_p, :], dtype=tf.float32)
            n = tf.convert_to_tensor(images_train[idxs_n, :], dtype=tf.float32)
        return a,p,n
