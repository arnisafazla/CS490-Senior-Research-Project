import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
  
def define_generator(config):
  init = keras.initializers.RandomNormal(stddev=config['init_std'])
  in_label = layers.Input(shape=(1,), name='label_input')
  li = layers.Embedding(config['n_classes'], config['in_shape'][1], name='label_embedding')(in_label)
  li = layers.Reshape((-1,))(li)
  li = layers.RepeatVector(config['in_shape'][0], name='label_repeat')(li)

  in_lat = layers.Input(shape=(config['latent_dim'],), name='seq_input')
  lat = layers.Dense(config['in_shape'][0] * config['in_shape'][1], name='lat_upsample')(in_lat)
  lat = layers.Reshape((config['in_shape'][0], config['in_shape'][1]))(lat)
  merge = keras.layers.Concatenate(name='concatenate', axis=2)([li, lat])

  hidden1 = layers.LSTM(config['in_shape'][1], name='hidden1', return_sequences=True, kernel_initializer=init)(merge)
  out_layer = layers.LSTM(config['in_shape'][1], name='out_LSTM', return_sequences=True, kernel_initializer=init)(hidden1)
 
  model = keras.Model([in_label, in_lat], out_layer, name='generator')
  return model