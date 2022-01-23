# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import random

main_dir = os.getcwd()
sys.path.append(os.path.join(main_dir, 'PyMO'))
from pymo import parsers
from pymo import viz_tools
from pymo import preprocessing
from pymo import data as mocapdata

class Dataset(object):
  def __init__(
                self, 
                emotions,
                path,
                # path to drive will be constant, go to the folder named the emotion and load all the files in it.
                step_size = 20,
                frames = 400
    ):
    self.emotions = np.array(emotions).reshape((-1))
    self.step_size = step_size
    self.onehotencoder = OneHotEncoder()
    self.onehotencoder.fit_transform(self.emotions.reshape(-1,1))
    # to encode: self.onehotencoder.transform([['joy']]).toarray().reshape((-1))
    # to decode: self.onehotencoder.inverse_transform([[1,0,0]])

    self.ordinalencoder = OrdinalEncoder()
    self.ordinalencoder.fit_transform(self.emotions.reshape((-1,1)))
    # to encode: self.ordinalencoder.transform(emotions.reshape((-1,1)))
    # to decode: self.ordinalencoder.inverse_transform([[2]])
    self.samples = np.zeros(self.emotions.shape[0])
    self.Y_vec = []
    self.X = []
    self.Y_ord = []
    self.data = mocapdata.MocapData()
    self.feature_names = []  
    self.position_features = ['Hips_Xposition', 'Hips_Yposition', 'Hips_Zposition']
    self.path = path
    self.frames = frames
    self.__load_data__()

  def __load_data__(self):
    parser = parsers.BVHParser()
    for emotion in self.emotions:
      data_path = self.path + '/' + str(emotion)
      file_names = os.listdir(data_path)
      print(emotion)
      one_hot_encoded_emotion = self.onehotencoder.transform([[emotion]]).toarray().reshape((-1))
      ordinal_encoded_emotion = self.ordinalencoder.transform([[emotion]]).reshape((-1))
      for file_name in tqdm(file_names):
        file_path = data_path + '/' + file_name
        parser.parse(file_path)
        parser.data.values = parser.data.values[1:]  
        length = len(parser.data.values)
        no_of_parts = (length - self.frames) // self.step_size
        for i in range(no_of_parts):
          sample = parser.data.values[i * self.step_size:i * self.step_size + self.frames]
          self.X.append(np.array(sample.drop(columns=self.position_features)))
          if np.array(sample.drop(columns=self.position_features)).shape == (200,72):
            print(file_name)
          self.Y_vec.append(one_hot_encoded_emotion)
          self.Y_ord.append(ordinal_encoded_emotion)

    self.data = parser.data
    self.feature_names = parser.data.values.columns   #.drop(self.position_features)
    
    self.X = np.array(self.X)
    self.Y_vec = np.array(self.Y_vec)
    self.Y_ord = np.array(self.Y_ord)

  @staticmethod
  def classes_dist(dataset):
    size = dataset.X.shape[0]
    index = [*range(size)]
    random.shuffle(index)
    X_shuffled = np.array([dataset.X[i] for i in index])
    Y_shuffled_ord = np.array([dataset.Y_ord[i] for i in index])
    Y_shuffled_vec = np.array([dataset.Y_vec[i] for i in index])
    dataset.X_samples = []
    dataset.Y_ord_samples = []
    dataset.Y_vec_samples = []
    for i in range(dataset.emotions.shape[0]):
      take = (Y_shuffled_ord == i).reshape(-1,)
      # return X_shuffled, take
      dataset.X_samples.append(X_shuffled[take])
      dataset.Y_ord_samples.append(Y_shuffled_ord[take])
      dataset.Y_vec_samples.append(Y_shuffled_vec[take])
    return [row.shape[0] for row in dataset.X_samples]

  @staticmethod
  def viz_dist(dataset):
    dist = dataset.classes_dist(dataset)
    x = np.arange(len(dist))
    y = dist
    LABELS = dataset.ordinalencoder.categories_[0][:8]
    plt.bar(x, y, align='center')
    plt.xticks(x, LABELS)
    return plt.figure
  

  @staticmethod
  def balance(dataset):
    size = dataset.X.shape[0]
    index = [*range(size)]
    random.shuffle(index)
    X_shuffled = np.array([dataset.X[i] for i in index])
    Y_shuffled_ord = np.array([dataset.Y_ord[i] for i in index])
    Y_shuffled_vec = np.array([dataset.Y_vec[i] for i in index])
    dataset.X_samples = []
    dataset.Y_ord_samples = []
    dataset.Y_vec_samples = []
    for i in range(dataset.emotions.shape[0]):
      take = (Y_shuffled_ord == i).reshape(-1,)
      dataset.X_samples.append(X_shuffled[take])
      dataset.Y_ord_samples.append(Y_shuffled_ord[take])
      dataset.Y_vec_samples.append(Y_shuffled_vec[take])
    min = np.min([row.shape[0] for row in dataset.X_samples])
    print('No. of samples in each class will be: ', min)
    dataset.X = np.concatenate([row[:min] for row in dataset.X_samples],axis=0)
    dataset.Y_ord = np.concatenate([row[:min] for row in dataset.Y_ord_samples],axis=0)
    dataset.Y_vec = np.concatenate([row[:min] for row in dataset.Y_vec_samples],axis=0)
    

  def train_test_split(self, test_size = 0.33, ord = False):
    size = self.X.shape[0]
    index = [*range(size)]
    random.shuffle(index)
    X_shuffled = np.array([self.X[i] for i in index])
    if ord:
      Y_shuffled = np.array([self.Y_ord[i] for i in index])
    else:
      Y_shuffled = np.array([self.Y_vec[i] for i in index])
    
    split = size - int(size * test_size)
    return X_shuffled[0:split], Y_shuffled[0:split], X_shuffled[split:], Y_shuffled[split:]

  def generate_real_samples(self, n_samples):
    seq, labels = self.X, self.Y_ord
    r = np.random.randint(0, seq.shape[0], n_samples)
    X, labels = seq[r]/180, labels[r] 
    y = -np.ones((n_samples, 1))
    return [labels, X], y

  def generate_fake_samples(self, n_samples):
    seq, labels = self.X, self.Y_ord
    r = np.random.randint(0, seq.shape[0], n_samples)
    X, labels_tmp = seq[r]/180, labels[r].reshape((-1,))
    l = np.random.randint(1, self.emotions.shape[0], n_samples) 
    # randomly change to class labels to another
    labels_tmp = (labels_tmp + l) % self.emotions.shape[0]
    y = np.ones((n_samples, 1))
    return [labels_tmp.reshape((-1,1)), X], y

  def transform(self, rotation_data):
    rotation_data = rotation_data * 180
    pos_values = np.zeros((rotation_data.shape[1], 3))
    full_values = np.array([np.concatenate((pos_values,sample), axis=1) for sample in rotation_data])
    values = [pd.DataFrame(data=sample, columns=self.feature_names) for sample in full_values]
    mocap = np.repeat([self.data.clone()], rotation_data.shape[0])
    for i in range(rotation_data.shape[0]):
      mocap[i].values = values[i]
    parametrizer = preprocessing.MocapParameterizer(param_type='position')
    position_transformed = parametrizer.transform(mocap)
    return position_transformed
