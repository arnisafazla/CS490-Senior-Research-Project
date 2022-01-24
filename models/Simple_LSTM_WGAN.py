import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import datetime, os
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model

main_dir = os.getcwd()
sys.path.append(os.path.join(main_dir, 'models'))
import Base_WGAN

class Simple_LSTM_WGAN(Base_WGAN):
  def __init__(self, models_dir, name, config, model_load=None):
    self.models_dir = models_dir
    if model_load == None:
      self.config = config
      self.const = ClipConstraint(self.config['clip'])
      self.latent_dim = self.config['latent_dim']
      self.in_shape = self.config['in_shape']
      self.n_classes = self.config['n_classes']
      self.init = keras.initializers.RandomNormal(stddev=self.config['init_std'])
      if self.config['critic_opt'] == 'RMS':
        self.critic_opt = keras.optimizers.RMSprop(learning_rate=self.config['critic_lr'])
      else:
        self.critic_opt = keras.optimizers.Adam(learning_rate=self.config['critic_lr'])
      if self.config['gan_opt'] == 'RMS':
        self.gan_opt = keras.optimizers.RMSprop(learning_rate=self.config['gan_lr'])
      else:
        self.gan_opt = keras.optimizers.Adam(learning_rate=self.config['gan_lr'])
      self.n_critic = self.config['n_critic']
      self.critic = self.define_critic()
      self.generator = self.define_generator()
      self.model = self.define_gan(self.generator, self.critic)
      self.start_point = 0
      self.folder_name = name + datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
      self.model_dir = os.path.join(self.models_dir, self.folder_name)
      self.metrics = [list() for i in range(4)]

    else:
      self.model = keras.models.load_model(os.path.join(model_load, 'model.h5'), 
              custom_objects={'ClipConstraint':ClipConstraint, 'wasserstein_loss':self.wasserstein_loss})
      self.generator = keras.models.load_model(os.path.join(model_load, 'generator.h5'))
      self.critic = keras.models.load_model(os.path.join(model_load, 'critic.h5'), 
              custom_objects={'ClipConstraint':ClipConstraint, 'wasserstein_loss':self.wasserstein_loss})
      self.critic.trainable = False
      self.start_point = int(re.match('.*?([0-9]+)$', model_load).group(1))
      self.folder_name = model_load[0:-len(os.path.basename(model_load))]
      self.model_dir = os.path.join(self.models_dir, self.folder_name)
      with open(os.path.join(model_load, 'metrics.txt')) as file:
        self.metrics = json.load(file)
      with open(os.path.join(self.model_dir, 'config.json')) as file:
        self.config = json.load(file)
      self.const = ClipConstraint(self.config['clip'])
      self.latent_dim = self.config['latent_dim']
      self.in_shape = self.config['in_shape']
      self.n_classes = self.config['n_classes']
      self.n_critic = self.config['n_critic']

  def define_critic(self):
    init = keras.initializers.RandomNormal(stddev=init_std)
    in_label = layers.Input(shape=(1,), name='label_input')
    # embedding for categorical input
    li = layers.Embedding(self.n_classes, self.in_shape[1], name='label_embedding')(in_label)  # had 50 nodes originally
    # upsample to 69
    # li = layers.Dense(self.in_shape[1], name='label_upsample', kernel_constraint=self.const, kernel_initializer=self.init)(li)
    # repeat the categorical input by no. of frames
    li = layers.Reshape((-1,))(li)
    li = layers.RepeatVector(self.in_shape[0], name='label_repeat')(li)
    # li = layers.Reshape((li.shape[1], 1, li.shape[2]))(li)
    # input sequence (mocap data)
    in_seq = layers.Input(shape=self.in_shape, name='sequence_input')
    # seq = layers.Reshape((in_seq.shape[1], 1, in_seq.shape[2]))(in_seq)
    # concatenate as another dimension
    merge = layers.Concatenate(name='concatenate', axis=2)([li, in_seq])
    # hidden1 = layers.ConvLSTM1D(self.in_shape[1], 2, name='conv1', return_sequences=False, kernel_initializer=self.init, kernel_constraint=self.const)(merge)
    hidden1 = layers.LSTM(self.in_shape[1], name='lstm1', return_sequences=True, kernel_initializer=self.init, kernel_constraint=self.const)(merge)
    hidden1 = layers.BatchNormalization()(hidden1)
    # hidden1 = layers.Reshape((hidden1.shape[1], hidden1.shape[3]))(hidden1)
    hidden2 = layers.LSTM(hidden1.shape[2], name='lstm2', kernel_initializer=init, kernel_constraint=self.const)(hidden1)   
    hidden2 = layers.BatchNormalization()(hidden2)

    dense = layers.Dense(32, name='dense', activation='relu', kernel_initializer=self.init, kernel_constraint=self.const)(hidden2)
    out_layer = layers.Dense(1, activation='linear', name='out_layer', kernel_initializer=self.init, kernel_constraint=self.const)(dense)

    model = keras.Model([in_label, in_seq], out_layer, name='critic')
    model.compile(loss=self.wasserstein_loss, optimizer=self.critic_opt)
    return model

  # define the standalone generator model
  def define_generator(self):
    # label input
    in_label = layers.Input(shape=(1,), name='label_input')
    # embedding for categorical input
    li = layers.Embedding(self.n_classes, self.in_shape[1], name='label_embedding')(in_label)  # had 50 nodes originally
    # upsample to 69
    # li = layers.Dense(self.in_shape[1], name='label_upsample', kernel_initializer=self.init, )(li)
    # repeat the categorical input by no. of frames
    li = layers.Reshape((-1,))(li)
    li = layers.RepeatVector(self.in_shape[0], name='label_repeat')(li)
    # li = layers.Reshape((li.shape[1], 1, li.shape[2]))(li)

    # generator input
    in_lat = layers.Input(shape=(self.latent_dim,), name='seq_input')
    lat = layers.Dense(self.in_shape[0] * self.in_shape[1], name='lat_upsample')(in_lat)
    # lat = layers.Reshape((self.in_shape[0], 1, self.in_shape[1]))(lat)
    lat = layers.Reshape((self.in_shape[0], self.in_shape[1]))(lat)
    # lat = layers.LSTM(self.in_shape[1], name='lat_LSTM', return_sequences=True, kernel_initializer=self.init)(lat)
    # merge them
    merge = keras.layers.Concatenate(name='concatenate', axis=2)([li, lat])

    hidden1 = layers.LSTM(self.in_shape[1], name='hidden1', return_sequences=True, kernel_initializer=self.init)(merge)
    out_layer = layers.LSTM(self.in_shape[1], name='out_LSTM', return_sequences=True, kernel_initializer=self.init)(hidden1)
    
    # out_layer = layers.Reshape((self.in_shape[0], self.in_shape[1]))(out_layer)
    # define model
    model = keras.Model([in_label, in_lat], out_layer, name='generator')
    
    return model

  # define the combined generator and critic model, for updating the generator
  def define_gan(self, g_model, d_model):
    # make weights in the critic not trainable
    # for layer in d_model.layers:
     #  if not isinstance(layer, layers.BatchNormalization):
     #   layer.trainable = False
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_label, gen_noise = g_model.input
    # get seq output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to critic
    gan_output = d_model([gen_label, gen_output])
    # define gan model as taking noise and label and outputting a classification
    model = keras.Model([gen_label, gen_noise], gan_output)
    # compile model
    model.compile(loss=self.wasserstein_loss, optimizer=self.gan_opt)
    return model
    
  # use the generator to generate n fake examples, with class labels
  def generate_fake_samples(self, n_samples):
    # generate points in latent space
    labels_input, z_input = Tools.generate_latent_points(self.latent_dim, n_samples, self.n_classes)
    # predict outputs
    seq = self.generator.predict([labels_input, z_input])
    # create class labels
    y = np.ones((n_samples, 1))
    return [labels_input, tf.convert_to_tensor(seq, dtype=tf.float32)], tf.convert_to_tensor(y, dtype=tf.float32)

  # implementation of wasserstein loss
  @staticmethod
  def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
 
  def train(self, dataset, n_epochs=50, n_batch=64, n_critic=5, verbose=1, dir='/content/drive/MyDrive/CS490/GAN_logs'):
    if not os.path.exists(self.model_dir):
      os.mkdir(self.model_dir)
    with open(os.path.join(self.model_dir, 'config.json'), 'w') as file:
      json.dump(self.config, file)
    plot_model(self.critic, show_shapes=True, show_layer_names=True, to_file=os.path.join(self.model_dir, 'critic.png'))
    plot_model(self.generator, show_shapes=True, show_layer_names=True, to_file=os.path.join(self.model_dir, 'generator.png'))
    plot_model(self.model, show_shapes=True, show_layer_names=True, to_file=os.path.join(self.model_dir, 'model.png'))
    
    device_name = tf.test.gpu_device_name()
    X, Y = dataset.X, dataset.Y_ord
    if device_name == '/device:GPU:0':
      with tf.device('/device:GPU:0'):
        logdir = os.path.join(dir, self.folder_name)
        tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        tensorboard.set_model(self.critic)
        # TRAINING
        bat_per_epo = int(X.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        if self.config['wrong_labels']:
          other_batch = int(half_batch / 2)
        else:
          other_batch = half_batch
        # manually enumerate epochs
        for i in range(self.start_point, n_epochs):
          for j in range(bat_per_epo):
            d_loss_real, d_loss_fake, d_loss_wrong = list(), list(), list()
            for _ in range(self.n_critic):
              # train with real samples
              [labels_real, X_real], y_real = dataset.generate_real_samples(half_batch)
              d_tmp_real = self.critic.train_on_batch([labels_real, X_real], y_real)
              d_loss_real.append(d_tmp_real)
              if self.config['wrong_labels']:
                # train with real samples wrong class labels
                [labels_wrong, X_wrong], y_wrong = dataset.generate_fake_samples(other_batch)
                d_tmp_wrong = self.critic.train_on_batch([labels_wrong, X_wrong], y_wrong)
                d_loss_wrong.append(d_tmp_wrong)
              else:
                d_loss_wrong.append(0)
              # train with fake samples
              [labels_fake, X_fake], y_fake = self.generate_fake_samples(other_batch)
              d_tmp_fake = self.critic.train_on_batch([labels_fake, X_fake], y_fake)
              d_loss_fake.append(d_tmp_fake)
            # prepare points in latent space as input for the generator
            [labels_input, z_input] = Tools.generate_latent_points(self.latent_dim, n_batch, self.n_classes)
            # create inverted labels for the fake samples
            y_gan = tf.convert_to_tensor(-np.ones((n_batch, 1)))
            # update the generator via the critic's error
            g_loss_batch = self.model.train_on_batch([labels_input, z_input], y_gan)
            # summarize loss on this batch"""
            # g_loss_batch = 0
            self.metrics[0].append(mean(d_loss_fake))
            self.metrics[1].append(mean(d_loss_real))
            self.metrics[2].append(mean(d_loss_wrong))
            self.metrics[3].append(g_loss_batch)  
            if verbose == 1 or verbose == 2:
              print('>%d, %d/%d, d_loss_fake=%.3f, d_loss_real=%.3f, d_loss_wrong=%.3f, g=%.3f' %(i+1, j+1, bat_per_epo, mean(d_loss_fake), mean(d_loss_real), mean(d_loss_wrong), g_loss_batch))
          logs = [mean(d_loss_fake), mean(d_loss_real), mean(d_loss_wrong), g_loss_batch]
          names = ["d_loss_fake", "d_loss_real", "d_loss_wrong", "g_loss_batch"]
          tensorboard.on_epoch_end(i+1, Tools.named_logs(self.model, names, logs))
          epoch_dir = os.path.join(self.model_dir, 'epoch_' + str(i+1))
          os.mkdir(epoch_dir)
          self.critic.trainable = False
          self.model.save(os.path.join(epoch_dir, 'model.h5'))
          self.generator.save(os.path.join(epoch_dir, 'generator.h5'))
          self.critic.trainable = True
          self.critic.save(os.path.join(epoch_dir, 'critic.h5'))
          self.critic.trainable = False
          with open(os.path.join(epoch_dir, 'metrics.txt'), 'w') as file:
            json.dump(self.metrics, file)
          if verbose == 2:
            self.save_checkpoint(dataset, epoch_dir, n_samples=1)
    else:
      with tf.device('/cpu:0'):
        print('Not connected to GPU')
    return metrics

  # generate n_samples per label and save as images
  def save_checkpoint(self, dataset, dir, n_samples=3):
    # prepare fake examples
    labels = np.arange(self.n_classes)
    labels = np.repeat(labels, n_samples)
    outputs = self.generate(labels)
    # visualize and plot poses
    position_transformed = []
    for i in range(outputs.shape[0]):
      position_transformed.append(dataset.transform(np.array([outputs[i]]))[0])
    for i, mocap_track in enumerate(position_transformed):
      fig = Tools.stickfigure(mocap_track, step=20, rows=5, title=dataset.ordinalencoder.inverse_transform([[labels[i]]]), figsize=(8,8))
      fig.savefig(os.path.join(dir, str(i) + '.png'))
      plt.close()

  # create a line plot of loss for the gan and save to file
  @staticmethod
  def plot_history(metrics):
    # plot loss
    loss_fake, loss_real, loss_wrong, g_loss = metrics
    plt.subplot(2, 1, 1)
    plt.plot(loss_real, label='d_loss_real')
    plt.plot(loss_fake, label='d_loss_fake')
    # plt.plot(loss_wrong, label='d_loss_wrong')
    plt.plot(g_loss, label='generator_loss')
    plt.legend()
    return plt.figure
 
  def generate(self, labels):
    [labels_input, z_input] = Tools.generate_latent_points(self.latent_dim, labels.shape[0], self.n_classes)
    # print(z_input)
    outputs = self.generator.predict([labels, z_input])
    return outputs