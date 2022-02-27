import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
    
def define_norm_generator(config):
  init = keras.initializers.RandomNormal(stddev=config['init_std'])
  in_label = layers.Input(shape=(1,), name='label_input')
  if config['generator_batch_norm'] or config['generator_layer_norm']:
    li = in_label
  else:
    li = layers.CategoryEncoding(num_tokens=config['n_classes'], output_mode="one_hot", name='one-hot')(in_label)
  li = layers.RepeatVector(config['in_shape'][0], name='repeat')(li)

  in_lat = layers.Input(shape=(config['latent_dim'],), name='seq_input')
  lat = layers.Dense( config['in_shape'][0] * config['in_shape'][1], name='lat_upsample')(in_lat)
  lat = layers.Reshape((config['in_shape'][0], config['in_shape'][1]))(lat)
  # merge = keras.layers.Concatenate(name='concatenate', axis=2)([li, lat])

  hidden1 = layers.LSTM(config['in_shape'][1]*2, name='LSTM1', return_sequences=True, kernel_initializer=init)(lat)
  merged = layers.Concatenate(axis=2, name='concatenate')([hidden1, li])
  if config['generator_batch_norm']:
    merged = layers.TimeDistributed(ConditionalBatchNorm(n_classes=config['n_classes'], name='conditional_batch_norm'))(merged)
  elif config['generator_layer_norm']:
    merged = layers.TimeDistributed(ConditionalLayerNorm(n_classes=config['n_classes'], name='conditional_layer_norm'))(merged)
  elif config['generator_layer_norm_plus']:
    merged = layers.TimeDistributed(ConditionalLayerNormPlus(n_classes=config['n_classes'], name='conditional_layer_norm_plus'))(merged)
  hidden2 = layers.LSTM(config['in_shape'][1], name='out_LSTM', return_sequences=True, kernel_initializer=init)(merged)
  merged2 = layers.Concatenate(axis=2, name='concatenate2')([hidden2, li])
  if config['generator_batch_norm']:
    merged2 = layers.TimeDistributed(ConditionalBatchNorm(n_classes=config['n_classes'], name='conditional_batch_norm2'))(merged2)
  elif config['generator_layer_norm']:
    merged2 = layers.TimeDistributed(ConditionalLayerNorm(n_classes=config['n_classes'], name='conditional_layer_norm2'))(merged2)
  elif config['generator_layer_norm_plus']:
    merged2 = layers.TimeDistributed(ConditionalLayerNormPlus(n_classes=config['n_classes'], name='conditional_layer_norm_plus2'))(merged2)
  model = keras.Model([in_label, in_lat], merged2, name='generator')
  return model

class ConditionalBatchNorm(layers.Layer):
  def __init__(self, n_classes, **kwargs):
    super().__init__(**kwargs)
    self.n_classes = n_classes
  def build(self, input_shape):
    self.seq_len = input_shape[1] - 1
    self.gamma = self.add_weight(shape=[self.n_classes, self.seq_len], 
        initializer='zeros', trainable=True, name='gamma')
    self.beta = self.add_weight(shape=[self.n_classes, self.seq_len], 
        initializer='zeros', trainable=True, name='beta')
    self.moving_mean = self.add_weight(shape=[1, self.seq_len],
        initializer='zeros', trainable=False, name='moving_mean')
    self.moving_var = self.add_weight(shape=[1, self.seq_len], 
        initializer='zeros', trainable=False, name='moving_var')
    self.alpha = 0.99  # alpha is the decay parameter for exponential moving average
    # it is 0.99 in keras.layers.BatchNormalization so I use it too.
    self.eps = 0.00001  # only for prevent dividing by 0. keras.layers.BatchNormalization use 0.001, I use smaller for safer
  def call(self, inputs, training=False):
    x, labels = tf.split(inputs, [self.seq_len, 1], axis=1)
    labels = tf.cast(labels, tf.int32)
    beta = tf.gather(self.beta, labels)
    # print(beta)
    beta = tf.reshape(beta, (-1, beta.shape[-1]))
    # print(beta)
    gamma = tf.gather(self.gamma, labels)
    gamma = tf.reshape(gamma, (-1, gamma.shape[-1]))
    if training:
      mean, var = tf.nn.moments(x, axes=(0), keepdims=True)
      self.moving_mean.assign(self.alpha * self.moving_mean + (1-self.alpha)*mean)
      self.moving_var.assign(self.alpha * self.moving_var + (1-self.alpha)*var)
      # mean = tf.repeat(mean, labels.shape[0], axis=0)
      # var = tf.repeat(var, labels.shape[0], axis=0)
      # print(x.shape, mean.shape, var.shape, beta.shape, gamma.shape)
      output = tf.nn.batch_normalization(x, mean, var, beta, gamma, self.eps)
    else:
      output = tf.nn.batch_normalization(x, self.moving_mean, self.moving_var, beta, gamma, self.eps)
    return output
  def compute_output_shape(self, input_shape):
    return (None, self.seq_len)
  def get_config(self):
    config = super().get_config()
    config.update({
        "n_classes": self.n_classes
    })
    return config

# Conditional layer norm but calculate beta and gammas separately using MLPs.
class ConditionalLayerNormPlus(layers.Layer):
  def __init__(self, n_classes, **kwargs):
    super().__init__(**kwargs)
    self.n_classes = n_classes
  def build(self, input_shape):
    self.seq_len = input_shape[1] - self.n_classes
    # self.gamma_embedding = layers.Embedding(1, self.seq_len, name='gamma_embedding')
    self.gamma_dense1 = layers.Dense(int(self.seq_len / 2), name='gamma_dense1')
    self.gamma_dense2 = layers.Dense(self.seq_len, name='gamma_dense2')
    # self.beta_embedding = layers.Embedding(1, self.seq_len, name='beta_embedding')
    self.beta_dense1 = layers.Dense(int(self.seq_len / 2), name='beta_dense1')
    self.beta_dense2 = layers.Dense(self.seq_len, name='beta_dense2')
    self.reshape = layers.Reshape((-1,))
    self.layer_norm = layers.LayerNormalization(center=False, scale=False)
    self.eps = 0.00001  # only for prevent dividing by 0. keras.layers.BatchNormalization use 0.001, I use smaller for safer
  def call(self, inputs, training=False):
    x, labels = tf.split(inputs, [self.seq_len, self.n_classes], axis=1)
    # print(x.shape, labels.shape)
    # gamma = self.gamma_embedding(labels)
    # gamma = self.reshape(gamma)
    gamma = self.gamma_dense1(labels)
    gamma = self.gamma_dense2(gamma)
    # beta = self.beta_embedding(labels)
    # beta = self.reshape(beta)
    beta = self.beta_dense1(labels)
    beta = self.beta_dense2(beta)
    # print(gamma.shape, beta.shape)
    output = tf.math.add(tf.math.multiply(self.layer_norm(x), gamma), beta)
    return output
  def compute_output_shape(self, input_shape):
    return (None, self.seq_len)
  def get_config(self):
    config = super().get_config()
    config.update({
        "n_classes": self.n_classes
    })
    return config

class ConditionalLayerNorm(layers.Layer):
  def __init__(self, n_classes, **kwargs):
    super().__init__(**kwargs)
    self.n_classes = n_classes
  def build(self, input_shape):
    self.seq_len = input_shape[1] - 1
    self.gamma = self.add_weight(shape=[self.n_classes, self.seq_len], 
        initializer='zeros', trainable=True, name='gamma')
    self.beta = self.add_weight(shape=[self.n_classes, self.seq_len], 
        initializer='zeros', trainable=True, name='beta')
    self.eps = 0.00001  # only for prevent dividing by 0. keras.layers.BatchNormalization use 0.001, I use smaller for safer
  def call(self, inputs, training=False):
    x, labels = tf.split(inputs, [self.seq_len, 1], axis=1)
    labels = tf.cast(labels, tf.int32)
    beta = tf.gather(self.beta, labels)
    beta = tf.reshape(beta, (-1, beta.shape[-1]))
    gamma = tf.gather(self.gamma, labels)
    gamma = tf.reshape(gamma, (-1, gamma.shape[-1]))
    mean, var = tf.nn.moments(x, axes=(1), keepdims=True)
    output = tf.nn.batch_normalization(x, mean, var, beta, gamma, self.eps)
    return output
  def compute_output_shape(self, input_shape):
    return (None, self.seq_len)
  def get_config(self):
    config = super().get_config()
    config.update({
        "n_classes": self.n_classes
    })
    return config