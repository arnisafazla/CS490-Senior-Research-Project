# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf

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

    self.n_features = self.X.shape[2]

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
    
  def get_size(self):
      return self.X.shape[0]

  @staticmethod
  def smoothen(rotation_data):
    sin = np.sin(np.deg2rad(rotation_data))
    cos = np.cos(np.deg2rad(rotation_data))
    return np.concatenate((sin, cos), axis=2)

  @staticmethod
  def atan(rotation_data):
    sin = rotation_data[:,:,:int(rotation_data.shape[2]/2)]
    cos = rotation_data[:,:,int(rotation_data.shape[2]/2):]
    return np.arctan2(sin, cos) * 180 / np.pi

  # For the simple LSTM model called Classifier can check if Classifier is better with smoothened data
  def train_test_split(self, test_size = 0.33, ord = False, smoothen=False):
    size = self.X.shape[0]
    index = [*range(size)]
    random.shuffle(index)
    X_shuffled = np.array([self.X[i] for i in index])
    if smoothen:
      X_shuffled = self.smoothen(X_shuffled)
    if ord:
      Y_shuffled = np.array([self.Y_ord[i] for i in index])
    else:
      Y_shuffled = np.array([self.Y_vec[i] for i in index])
    
    split = size - int(size * test_size)
    return X_shuffled[0:split], Y_shuffled[0:split], X_shuffled[split:], Y_shuffled[split:]

  def generate_real_samples(self, n_samples, smoothen=True):
    seq, labels = self.X, self.Y_ord
    r = np.random.randint(0, seq.shape[0], n_samples)
    labels = labels[r]
    if smoothen:
      X = self.smoothen(seq[r])
    else:
      X = seq[r]/180
    y = -np.ones((n_samples, 1))
    return [tf.convert_to_tensor(labels), tf.convert_to_tensor(X, dtype=tf.dtypes.float32)], tf.convert_to_tensor(y)

  # not necessary anymore?
  def generate_fake_samples(self, n_samples, smoothen=True):
    seq, labels = self.X, self.Y_ord
    r = np.random.randint(0, seq.shape[0], n_samples)
    labels_tmp = labels[r].reshape((-1,))
    if smoothen:
      X = self.smoothen(seq[r])
    else:
      X = seq[r]/180
    l = np.random.randint(1, self.emotions.shape[0], n_samples) 
    # randomly change to class labels to another
    labels_tmp = (labels_tmp + l) % self.emotions.shape[0]
    y = np.ones((n_samples, 1))
    return [labels_tmp.reshape((-1,1)), X], y

  def transform(self, rotation_data, smoothen=True):
    if smoothen:
      rotation_data = self.atan(rotation_data)
    else:
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

  @staticmethod
  def stickfigure(mocap_track, title='', step=1, cols=2, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
    n = mocap_track.values.shape[0] // step
    
    fig, axs = plt.subplots(ncols=cols, nrows=n // cols , figsize=figsize, constrained_layout=True)
    for row in range(n // cols):
      for col in range(cols):    
        if joints is None:
            joints_to_draw = mocap_track.skeleton.keys()
        else:
            joints_to_draw = joints    
        if data is None:
            df = mocap_track.values
        else:
            df = data  
        frame = (row * cols + col) * step
        for joint in joints_to_draw:
            axs[row, col].scatter(x=df['%s_Xposition'%joint][frame], 
                        y=df['%s_Yposition'%joint][frame],  
                        alpha=0.6, c='b', marker='o')
            parent_x = df['%s_Xposition'%joint][frame]
            parent_y = df['%s_Yposition'%joint][frame]        
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]        
            for c in children_to_draw:
                child_x = df['%s_Xposition'%c][frame]
                child_y = df['%s_Yposition'%c][frame]
                axs[row, col].plot([parent_x, child_x], [parent_y, child_y], 'k-', lw=2)            
            if draw_names:
                axs[row, col].annotate(joint, 
                        (df['%s_Xposition'%joint][frame] + 0.1, 
                          df['%s_Yposition'%joint][frame] + 0.1))
            axs[row, col].axis('off')
    fig.suptitle(title)
    return fig

  # Giving an error
  @staticmethod
  def stickfigure3d(mocap_track, step=1, cols=2, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
    from mpl_toolkits.mplot3d import Axes3D
    n = 100 // step
    fig, axs = plt.subplots(ncols=cols, nrows=n // cols , figsize=figsize, constrained_layout=True)
    for row in range(n // cols):
      for col in range(cols):    
        if joints is None:
            joints_to_draw = mocap_track.skeleton.keys()
        else:
            joints_to_draw = joints  
        if data is None:
            df = mocap_track.values
        else:
            df = data     
        frame = (row * cols + col) * step
        for joint in joints_to_draw:
            parent_x = df['%s_Xposition'%joint][frame]
            parent_y = df['%s_Zposition'%joint][frame]
            parent_z = df['%s_Yposition'%joint][frame]
            # ^ In mocaps, Y is the up-right axis
            print(parent_x, parent_y, parent_z)
            axs[row, col].scatter(xs=parent_x, 
                        ys=parent_y,  
                        zs=parent_z,  
                        alpha=0.6, c='b', marker='o')        
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]    
            for c in children_to_draw:
                child_x = df['%s_Xposition'%c][frame]
                child_y = df['%s_Zposition'%c][frame]
                child_z = df['%s_Yposition'%c][frame]
                # ^ In mocaps, Y is the up-right axis

                axs[row, col].plot([parent_x, child_x], [parent_y, child_y], [parent_z, child_z], 'k-', lw=2, c='black')       
            if draw_names:
                axs[row, col].text(x=parent_x + 0.1, 
                        y=parent_y + 0.1,
                        z=parent_z + 0.1,
                        s=joint,
                        color='rgba(0,0,0,0.9)')
            axs[row, col].axis('off')
    return ax
