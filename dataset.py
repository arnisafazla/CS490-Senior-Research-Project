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
sys.path.append(main_dir)
from pymo import parsers
from pymo import viz_tools
from pymo import preprocessing
from pymo import data as mocapdata
from tools import Tools

# representation can be '6d', 'rot' or 'eul'.
# the data is kept as tf Tensors if representation == '6d'
class Dataset(object):
  def __init__(
                self, 
                emotions,
                path,
                # path to drive will be constant, go to the folder named the emotion and load all the files in it.
                step_size = 20,
                frames = 200,
                validation = 0,
                representation = '6d',
                data = None
    ):
    assert validation >= 0 and validation < 1
    assert representation in ['6d', 'rot', 'eul'], "representation is {s}".format(s=representation)
    self.emotions = np.array(emotions).reshape((-1))
    self.step_size = step_size
    self.representation = representation
    self.onehotencoder = OneHotEncoder()
    self.onehotencoder.fit_transform(self.emotions.reshape(-1,1))
    # to encode: self.onehotencoder.transform([['joy']]).toarray().reshape((-1))
    # to decode: self.onehotencoder.inverse_transform([[1,0,0]])

    self.ordinalencoder = OrdinalEncoder()
    self.ordinalencoder.fit_transform(self.emotions.reshape((-1,1)))
    # to encode: self.ordinalencoder.transform(emotions.reshape((-1,1)))
    # to decode: self.ordinalencoder.inverse_transform([[2]])
    self.Y_vec = []
    self.X = []
    self.Y_ord = []
    self.validation = validation
    if self.validation > 0:
      self.Y_vec_val = []
      self.X_val = []
      self.Y_ord_val = []
    self.data = mocapdata.MocapData()
    self.feature_names = []  
    self.position_features = ['Hips_Xposition', 'Hips_Yposition', 'Hips_Zposition']
    self.rotation_features = ['Hips_Yrotation',
       'Hips_Xrotation', 'Hips_Zrotation', 'Chest_Yrotation',
       'Chest_Xrotation', 'Chest_Zrotation', 'Chest2_Yrotation',
       'Chest2_Xrotation', 'Chest2_Zrotation', 'Chest3_Yrotation',
       'Chest3_Xrotation', 'Chest3_Zrotation', 'Chest4_Yrotation',
       'Chest4_Xrotation', 'Chest4_Zrotation', 'Neck_Yrotation',
       'Neck_Xrotation', 'Neck_Zrotation', 'Head_Yrotation', 'Head_Xrotation',
       'Head_Zrotation', 'RightCollar_Yrotation', 'RightCollar_Xrotation',
       'RightCollar_Zrotation', 'RightShoulder_Yrotation',
       'RightShoulder_Xrotation', 'RightShoulder_Zrotation',
       'RightElbow_Yrotation', 'RightElbow_Xrotation', 'RightElbow_Zrotation',
       'RightWrist_Yrotation', 'RightWrist_Xrotation', 'RightWrist_Zrotation',
       'LeftCollar_Yrotation', 'LeftCollar_Xrotation', 'LeftCollar_Zrotation',
       'LeftShoulder_Yrotation', 'LeftShoulder_Xrotation',
       'LeftShoulder_Zrotation', 'LeftElbow_Yrotation', 'LeftElbow_Xrotation',
       'LeftElbow_Zrotation', 'LeftWrist_Yrotation', 'LeftWrist_Xrotation',
       'LeftWrist_Zrotation', 'RightHip_Yrotation', 'RightHip_Xrotation',
       'RightHip_Zrotation', 'RightKnee_Yrotation', 'RightKnee_Xrotation',
       'RightKnee_Zrotation', 'RightAnkle_Yrotation', 'RightAnkle_Xrotation',
       'RightAnkle_Zrotation', 'RightToe_Yrotation', 'RightToe_Xrotation',
       'RightToe_Zrotation', 'LeftHip_Yrotation', 'LeftHip_Xrotation',
       'LeftHip_Zrotation', 'LeftKnee_Yrotation', 'LeftKnee_Xrotation',
       'LeftKnee_Zrotation', 'LeftAnkle_Yrotation', 'LeftAnkle_Xrotation',
       'LeftAnkle_Zrotation', 'LeftToe_Yrotation', 'LeftToe_Xrotation',
       'LeftToe_Zrotation']
    self.path = path
    self.frames = frames
    if data == None:
      self.__load_data__()
    else:
      self.X, self.X_val, self.Y_ord, self.Y_ord_val, self.Y_vec, self.Y_vec_val = [data[key] for key in list(data.keys())[:6]]
      self.data, self.feature_names, self.n_features = [data[key] for key in list(data.keys())[6:9]]

  def __load_data__(self):
    parser = parsers.BVHParser()
    for emotion in self.emotions:
      data_path = self.path + '/' + str(emotion)
      file_names = os.listdir(data_path)
      print(emotion)
      one_hot_encoded_emotion = self.onehotencoder.transform([[emotion]]).toarray().reshape((-1))
      ordinal_encoded_emotion = self.ordinalencoder.transform([[emotion]]).reshape((-1))
      if self.validation > 0:
        val_split = int(np.ceil(len(file_names) * self.validation))
        val_files = file_names[-val_split:]
        file_names = file_names[:-val_split]
      for file_name in tqdm(file_names):
        file_path = data_path + '/' + file_name
        parser.parse(file_path)
        parser.data.values = parser.data.values[1:]  
        length = len(parser.data.values)
        no_of_parts = (length - self.frames) // self.step_size
        for i in range(no_of_parts):
          sample = parser.data.values[i * self.step_size:i * self.step_size + self.frames]
          arr = np.array(sample[self.rotation_features])
          self.X.append(arr)
          self.Y_vec.append(one_hot_encoded_emotion)
          self.Y_ord.append(ordinal_encoded_emotion)
      if self.validation > 0:
        for file_name in tqdm(val_files):
          file_path = data_path + '/' + file_name
          parser.parse(file_path)
          parser.data.values = parser.data.values[1:]  
          length = len(parser.data.values)
          no_of_parts = (length - self.frames) // self.step_size
          for i in range(no_of_parts):
            sample = parser.data.values[i * self.step_size:i * self.step_size + self.frames]
            arr = np.array(sample[self.rotation_features])
            self.X_val.append(arr)            
            self.Y_vec_val.append(one_hot_encoded_emotion)
            self.Y_ord_val.append(ordinal_encoded_emotion)

    self.data = parser.data
    self.feature_names = parser.data.values.columns   #.drop(self.position_features)
    if self.representation == '6d':
      self.X = Tools.rots_to_ort6d(Tools.euler_to_rots(np.array(self.X)))
    else:
      self.X = np.array(self.X)
    self.Y_vec = np.array(self.Y_vec)
    self.Y_ord = np.array(self.Y_ord)
    if self.validation > 0:
      if self.representation == '6d':
        self.X_val = Tools.rots_to_ort6d(Tools.euler_to_rots(np.array(self.X_val)))
      else:
        self.X_val = np.array(self.X_val)
      self.Y_vec_val = np.array(self.Y_vec_val)
      self.Y_ord_val = np.array(self.Y_ord_val)

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

  # tracks x frames x joints x ?
  def convert_representation(self, X, rep='6d'):
    assert rep in ['6d', 'rot', 'eul'], "representation is {s}".format(s=repr)
    if self.representation == '6d':
      if rep == '6d':
        X = X.numpy()
      elif rep == 'rot':
        X = Tools.ort6d_to_rots(X)
    elif self.representation == 'rot':
      if rep == '6d':
        X = Tools.rots_to_ort6d(X).numpy()
      elif rep == 'rot':
        pass
    elif self.representation == 'eul':
      if rep == '6d':
        X = Tools.rots_to_ort6d(Tools.euler_to_rots(X)).numpy()
      elif rep == 'rot':
        X = Tools.euler_to_rots(X)
      elif rep == 'eul':
        pass
    return X.reshape((X.shape[0], X.shape[1], -1))

  # For the simple LSTM model called Classifier
  def train_test_split(self, test_size = 0.33, ord = False, rep='6d'):
    size = self.X.shape[0]
    index = [*range(size)]
    random.shuffle(index)
    X_shuffled = np.array([self.X[i] for i in index])
    if ord:
      Y_shuffled = np.array([self.Y_ord[i] for i in index])
    else:
      Y_shuffled = np.array([self.Y_vec[i] for i in index])
    split = size - int(size * test_size)
    X = self.convert_representation(X_shuffled, rep)
    return  X[0:split], Y_shuffled[0:split], X[split:], Y_shuffled[split:]

  def generate_real_samples(self, n_samples, val=False, rep='6d'):
    if val:
      seq, labels = self.X_val, self.Y_ord_val
    else:
      seq, labels = self.X, self.Y_ord
    r = np.random.randint(0, seq.shape[0], n_samples)
    labels = labels[r]
    X = seq[r]
    X = self.convert_representation(X, rep)
    y = -np.ones((n_samples, 1))
    return [tf.convert_to_tensor(labels), tf.convert_to_tensor(X, dtype=tf.float32)], tf.convert_to_tensor(y)

  def generate_fake_samples(self, n_samples, val=False, rep='6d'):
    if val:
      seq, labels = self.X_val, self.Y_ord_val
    else:
      seq, labels = self.X, self.Y_ord
    r = np.random.randint(0, seq.shape[0], n_samples)
    labels_tmp = labels[r].reshape((-1,))
    X = seq[r]
    X = self.convert_representation(X, rep)
    l = np.random.randint(1, self.emotions.shape[0], n_samples) 
    # randomly change to class labels to another
    labels_tmp = (labels_tmp + l) % self.emotions.shape[0]
    y = np.ones((n_samples, 1))
    return [tf.convert_to_tensor(labels_tmp.reshape((-1,1))), tf.convert_to_tensor(X, dtype=tf.float32)], tf.convert_to_tensor(y)

  # given euler values (shape: tracks x frames x features)
  # return list of mocap tracks with position.
  def eul_to_pos(self, rotation_data):
    pos_values = np.zeros((rotation_data.shape[1], 3))
    full_values = np.array([np.concatenate((pos_values,sample), axis=1) for sample in rotation_data])
    values = [pd.DataFrame(data=sample, columns=self.feature_names) for sample in full_values]
    mocap = np.repeat([self.data.clone()], rotation_data.shape[0])
    for i in range(rotation_data.shape[0]):
      mocap[i].values = values[i]
    parametrizer = preprocessing.MocapParameterizer(param_type='position')
    position_transformed = parametrizer.transform(mocap)
    return position_transformed

  # given rotation matrices (shape: tracks x frames x joints x 3 x 3)
  # return list of mocap tracks with position.
  def rots_to_pos(self, rots, rep='6d'):
    assert rep in ['6d', 'rots']
    if rep == '6d':
      rots = Tools.ort6d_to_rots(rots)
    rots_tracks = Tools.rots_to_dict(rots, self)
    X = rots_tracks.copy()
    data = self.data
    '''Converts joints rotations in Euler angles to joint positions'''
    Q = []
    # track:  frames x no_of_joints x (3 x 3)
    for track in X:
      tmp = track.copy()
      channels = []
      titles = []
      # Create a new DataFrame to store the exponential map rep
      pos_df = pd.DataFrame(index=data.values.index[:self.frames])
      tree_data = {}

      for (joint_id, joint) in enumerate(data.traverse()):
        parent = data.skeleton[joint]['parent']
        tree_data[joint]=[
                              [], # to store the rotation matrix
                              []  # to store the calculated position
                            ] 

        pos_values = [[0,0,0] for f in range(self.frames)]
        if joint not in tmp.keys():
          tmp[joint] = np.array([[[1., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 1.]] for i in range(self.frames)])
        if joint == data.root_name:
          tree_data[joint][0] = tmp[joint]
          tree_data[joint][1] = pos_values
        else:
          # for every frame i, multiply this joint's rotmat to the rotmat of its parent
          tree_data[joint][0] = np.asarray([np.matmul(tmp[joint][i], tree_data[parent][0][i]) 
                                            for i in range(len(tree_data[parent][0]))])

          # add the position channel to the offset and store it in k, for every frame i
          k = np.asarray([data.skeleton[joint]['offsets'] for i in range(len(tree_data[parent][0]))])

          # multiply k to the rotmat of the parent for every frame i
          q = np.asarray([np.matmul(k[i], tree_data[parent][0][i]) 
                          for i in range(len(tree_data[parent][0]))])

          # add q to the position of the parent, for every frame i
          tree_data[joint][1] = np.asarray([np.add(q[i], tree_data[parent][1][i])
                                            for i in range(len(tree_data[parent][1]))])
          
        # Create the corresponding columns in the new DataFrame
        pos_df['%s_Xposition'%joint] = pd.Series(data=[e[0] for e in tree_data[joint][1]], index=pos_df.index)
        pos_df['%s_Yposition'%joint] = pd.Series(data=[e[1] for e in tree_data[joint][1]], index=pos_df.index)
        pos_df['%s_Zposition'%joint] = pd.Series(data=[e[2] for e in tree_data[joint][1]], index=pos_df.index)

      new_track = data.clone()
      new_track.values = pos_df
      Q.append(new_track)
    return Q

  @staticmethod
  def stickfigure(mocap_track, title='', step=20, cols=5, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
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
