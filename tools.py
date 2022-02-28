import logging
import numpy as np
import tensorflow as tf

class Tools(object):
  @staticmethod
  def generate_latent_points(latent_dim, n_samples, n_classes):
    # generate points in the latent space
    x_input = np.random.normal(size=latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [tf.convert_to_tensor(labels), tf.convert_to_tensor(z_input)]

  @staticmethod
  def named_logs(names, logs):
    result = {}
    for l in zip(names, logs):
      result[l[0]] = l[1]
    return result

class Metrics(object):
  @staticmethod
  def confusion_matrix(critic, n_classes, n_samples, dataset, smoothen=True):
    cm = np.zeros((n_classes,n_classes))
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
      for label in range(n_classes):
        for i in range(n_samples):
          [labels_real, X_real], y_real = dataset.generate_real_samples(1, smoothen)
          while int(labels_real.numpy().item()) != label:
            [labels_real, X_real], y_real = dataset.generate_real_samples(1, smoothen)
          with tf.device('/device:GPU:0'):
            probs = critic.predict([np.arange(n_classes), np.repeat(X_real, n_classes, axis=0)])
            cm[np.argmax(probs), int(labels_real.numpy().item())] += 1
      return cm.tolist()
    else:
      with tf.device('/cpu:0'):
        logging.error('Not connected to GPU')
  @staticmethod
  def critic_loss(real, fake):
      real_loss = tf.reduce_mean(real)
      fake_loss = tf.reduce_mean(fake)
      return fake_loss - real_loss
  @staticmethod
  def generator_loss(fake):
      return -tf.reduce_mean(fake)


