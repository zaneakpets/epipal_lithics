from base.base_data_loader import BaseDataLoader
import numpy as np
import keras
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from data_loader.preprocess_images import get_random_eraser, preprocess_input
import re
import csv

class CosFaceGenerator(keras.utils.Sequence,):
    'Generates data for Keras'
    def __init__(self, config, datagen_args, is_train= True, is_random_transform = True, n_channels=3):
        'Initialization'
        self.config = config
        self.dim = (config.model.img_height, config.model.img_width)
        self.keras_datagen = ImageDataGenerator(**datagen_args)
        self.batch_size = self.config.data_loader.batch_size
        self.num_of_classes = []
        self.isTrain = is_train
        self.is_random_transform = is_random_transform

        if self.isTrain:
            self.data_folder = config.data_loader.data_dir_train
        else:
            self.data_folder = config.data_loader.data_dir_valid

        self.class_list = [dI for dI in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, dI))]

        if self.config.model.is_use_prior_weights:
            # read prior class confusion matrix
            T = np.genfromtxt(self.config.model.classes_confusion_prior,delimiter=',')
            rows_sum = np.sum(T,axis=1)
            self.T = T / rows_sum[:, np.newaxis]
            #print(np.sum(self.T,axis=1))
            # normalize matrix so each row will represent probability
            #self.T = T / np.linalg.norm(T, ord=2, axis=1, keepdims=True)

        self.n_channels = n_channels
        self.images_path_list = []
        self.num_of_images = []
        self.total_labels = 0
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_of_images / self.batch_size))

    def __getitem__(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)


        # choose batch_size images and remove them from image list
        try:
            images_path = random.sample(self.images_path_list, self.batch_size)
        except:
            images_path = random.sample(self.images_path_list, len(self.images_path_list))
            self.on_epoch_end()
            print('resetting')

        # Initialization
        X = np.empty((len(images_path), *self.dim, self.n_channels),dtype=np.float32)
        labels_list = []
        priod_label_list = []
        site_label_list = []

        # generate data
        for i, file_path in enumerate(images_path):
            self.images_path_list.remove(file_path)
            split_path = file_path.split('/')

            X[i, ] = self.read_and_preprocess_images(file_path)

            label_ = split_path[-2]
            labels_list.append(self.class_list.index(label_))
            #priod_label_list.append(self.period_dict[label_])
            #site_label_list.append(self.site_dict[label_])


        # custom loss in keras requires the labels and predictions to be in the same size
        if self.config.model.is_use_prior_weights:
            labels = np.zeros((len(labels_list), int(self.config.model.embedding_dim)), dtype=np.float)
            labels_arr = np.array(labels_list, dtype=np.int32)
            labels[:,:self.config.data_loader.num_of_classes] = self.T[labels_arr,:]
        else:
            labels = np.zeros((len(labels_list), int(self.config.model.embedding_dim)), dtype=np.int)
            labels[:,0] = np.array(labels_list, dtype=np.int32)
            labels[:, 1] = np.array(labels_list, dtype=np.int32)
            labels[:, 2] = np.array(labels_list, dtype=np.int32)

        if self.config.model.num_of_outputs == 1:
            labels_out = labels
        elif self.config.model.num_of_outputs == 2:
            labels_out = [ labels, labels[:,:self.config.data_loader.num_of_classes]]
        elif self.config.model.type == "vgg_attention":
            g_out = np.zeros((len(labels_list), int(self.config.model.embedding_dim)), dtype=np.int16)
            alpha_out = np.zeros((len(labels_list), self.config.model.img_width, self.config.model.img_height), dtype=np.int16)
            labels_out = [g_out, labels, alpha_out]

        return X, labels_out


    def on_epoch_end(self):
        'initialize indicators arrays '
        self.images_path_list = []
        total_labels = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.data_folder):
            for file in f:
                if '.jpg' or 'JPG' or 'png' or 'PNG' in file:
                    self.images_path_list.append(os.path.join(r, file))
                    folders = r.split('/')
                    total_labels.append(folders[-1])


        random.shuffle(self.images_path_list)

        self.num_of_images = len(self.images_path_list)
        self.total_labels = np.asarray(total_labels)




    def read_and_preprocess_images(self, image_path):
        # read image from disk
        img = image.load_img(image_path, target_size=(self.dim[0], self.dim[1],3))
        img = image.img_to_array(img)

        if self.config.data_loader.is_use_cutOut and self.isTrain:
            preprocess_function = get_random_eraser(v_l=0, v_h=1)
        else:
            preprocess_function = preprocess_input


        img = preprocess_function(img)

        # random transform
        if self.is_random_transform:
            x = self.keras_datagen.random_transform(img)
        else:
            x = img

        return x




