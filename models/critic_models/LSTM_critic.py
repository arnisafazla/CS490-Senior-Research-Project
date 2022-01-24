import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils

def define_critic(config):
    print(config)
    init = keras.initializers.RandomNormal(stddev=config['init_std'])
    in_label = layers.Input(shape=(1,), name='label_input')
    li = layers.Embedding(config['n_classes'], config['in_shape'][1], name='label_embedding')(in_label)  # had 50 nodes originally
    li = layers.Reshape((-1,))(li)
    li = layers.RepeatVector(config['in_shape'][0], name='label_repeat')(li)

    in_seq = layers.Input(shape=config['in_shape'], name='sequence_input')
    merge = layers.Concatenate(name='concatenate', axis=2)([li, in_seq])

    hidden1 = layers.LSTM(config['in_shape'][1], name='lstm1', return_sequences=True, kernel_initializer=init)(merge)
    if config['critic_batch_norm']:
      hidden1 = layers.BatchNormalization()(hidden1)
    if config['critic_dropout'] > 0:
      hidden1 = layers.Dropout(config['critic_dropout'])(hidden1)
    hidden2 = layers.LSTM(hidden1.shape[2], name='lstm2', kernel_initializer=init)(hidden1)   
    if config['critic_batch_norm']:
      hidden2 = layers.BatchNormalization()(hidden2)
    if config['critic_dropout'] > 0:
      hidden2 = layers.Dropout(config['critic_dropout'])(hidden2)

    dense = layers.Dense(32, name='dense', activation='relu', kernel_initializer=init)(hidden2)
    out_layer = layers.Dense(1, activation='linear', name='out_layer', kernel_initializer=init)(dense)

    model = keras.Model([in_label, in_seq], out_layer, name='critic')
    return model