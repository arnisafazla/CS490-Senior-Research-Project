import os, sys
main_dir = os.getcwd()
sys.path.append(os.path.join(main_dir, 'models'))
os.chdir(main_dir)
# append a path to any directory that the tensorflow_addons is installed in.
sys.path.append('/content/drive/MyDrive/CS490')

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils

def define_projection_critic(config):
    init = keras.initializers.RandomNormal(stddev=config['init_std'])
    in_label = layers.Input(shape=(1,), name='label_input')
    li = layers.Embedding(config['n_classes'], config['in_shape'][1], name='label_embedding')(in_label)  # had 50 nodes originally
    li = layers.Reshape((-1,))(li)

    in_seq = layers.Input(shape=config['in_shape'], name='sequence_input')

    hidden1 = layers.LSTM(config['in_shape'][1], name='lstm1', return_sequences=True, kernel_initializer=init, unroll=True)(in_seq)
    if config['critic_batch_norm']:
      hidden1 = layers.BatchNormalization()(hidden1)
    if config['critic_instance_norm']:
      hidden1 = tfa.layers.InstanceNormalization()(hidden1)
    if config['critic_layer_norm']:
      hidden1 = layers.LayerNormalization(axis=1 , center=True , scale=True)(hidden1)
    if config['critic_weight_norm']:
      hidden1 = layers.WeightNormalization(axis=1 , center=True , scale=True)(hidden1)
    if config['critic_dropout'] > 0:
      hidden1 = layers.Dropout(config['critic_dropout'])(hidden1)
    hidden2 = layers.LSTM(hidden1.shape[2], name='lstm2', kernel_initializer=init, unroll=True)(hidden1)   
    if config['critic_batch_norm']:
      hidden2 = layers.BatchNormalization()(hidden2)
    if config['critic_instance_norm']:
      hidden2 = tfa.layers.InstanceNormalization()(hidden2)
    if config['critic_layer_norm']:
      hidden2 = layers.LayerNormalization(axis=1 , center=True , scale=True)(hidden2)
    if config['critic_weight_norm']:
      hidden2 = layers.WeightNormalization(axis=1 , center=True , scale=True)(hidden2)
    if config['critic_dropout'] > 0:
      hidden2 = layers.Dropout(config['critic_dropout'])(hidden2)

    dot = layers.Dot(axes=(1))([hidden2, li])

    # process sequence data more separately
    dense = layers.Dense(32, name='dense', activation='relu', kernel_initializer=init)(hidden2)
    out_layer = layers.Dense(1, activation='linear', name='out_layer', kernel_initializer=init)(dense)

    out = layers.Add()([dot, out_layer])
    model = keras.Model([in_label, in_seq], out, name='critic')
    return model