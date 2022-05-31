import logging, copy, math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
import tensorflow as tf

main_dir = os.getcwd()
sys.path.append(os.path.join(main_dir, 'PyMO'))

class Tools(object):
  @staticmethod
  def _from_euler(y):
    ca = np.cos(y[:,0] * np.pi / 180)
    cb = np.cos(y[:,1] * np.pi / 180)
    cg = np.cos(y[:,2] * np.pi / 180)
    sa = np.sin(y[:,0] * np.pi / 180)
    sb = np.sin(y[:,1] * np.pi / 180)
    sg = np.sin(y[:,2] * np.pi / 180)        
    return np.stack((np.stack((cg*cb+sa*sb*sg, sg*ca, -cg*sb+sg*sa*cb), axis=1),
                          np.stack((-sg*cb+cg*sa*sb, cg*ca, sg*sb+cg*sa*cb), axis=1),
                          np.stack((ca*sb, -sa, ca*cb), axis=1)), axis=1)

  @staticmethod
  # convert euler values (shape: tracks x frames x features) of YXZ order
  # to rotation matrices (shape: tracks x frames x joints x 3 x 3)
  def euler_to_rots(euler):
      shape = (euler.shape[0], euler.shape[1], int(euler.shape[2]/3), 3, 3)
      eul2 = euler.reshape(-1, 3)
      rots = Tools._from_euler(eul2).reshape(shape)
      return rots

  @staticmethod
  # given rotation matrices (shape: tracks x frames x joints x 3 x 3)
  # return them as list of python dictionaries (keys are joints)
  # data is mocap data
  def rots_to_dict(rots, dataset):
    joints = []
    data = dataset.data
    for joint in data.traverse():
      joints.append(joint)
    joints2 = []
    for joint in dataset.rotation_features:
      joints2.append(joint.split('_')[0])
    joints = np.array(joints2).reshape(-1,3)[:,0]
    rots_tracks = []
    for j in range(rots.shape[0]):
      rots_dict = {}
      for i in range(len(joints)):
        rots_dict[joints[i]] = rots[j,:,i,:,:]
      rots_tracks.append(rots_dict)
    return rots_tracks

  # Adapted from "On the continuity of rotation representations in neural networks"
  # Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li.
  # Conference on Neural Information Processing Systems (NeurIPS) 2019.
  @staticmethod
  def tf_rotation6d_to_matrix(r6d):
    """ Compute rotation matrix from 6D rotation representation.
      Implementation base on 
          https://arxiv.org/abs/1812.07035
      [Inputs]
          6D rotation representation (last dimension is 6)
      [Returns]
          flattened rotation matrix (last dimension is 9)
    """
    tensor_shape = r6d.get_shape().as_list()

    if not tensor_shape[-1] == 6:
      raise AttributeError("The last demension of the inputs in tf_rotation6d_to_matrix should be 6, \
          but found tensor with shape {}".format(tensor_shape[-1]))

    r6d   = tf.reshape(r6d, [-1,6])
    x_raw = r6d[:,0:3]
    y_raw = r6d[:,3:6]

    x = tf.nn.l2_normalize(x_raw, axis=-1)
    z = tf.linalg.cross(x, y_raw)
    z = tf.nn.l2_normalize(z, axis=-1)
    y = tf.linalg.cross(z, x)

    x = tf.reshape(x, [-1,3,1])
    y = tf.reshape(y, [-1,3,1])
    z = tf.reshape(z, [-1,3,1])
    matrix = tf.concat([x,y,z], axis=-1)

    if len(tensor_shape) == 1:
      matrix = tf.reshape(matrix, [9])
    else:
      output_shape = tensor_shape[:-1] + [9]
      matrix = tf.reshape(matrix, output_shape)

    return matrix

  @staticmethod
  def tf_matrix_to_rotation6d(mat):
    """ Get 6D rotation representation for rotation matrix.
      Implementation base on 
          https://arxiv.org/abs/1812.07035
      [Inputs]
          flattened rotation matrix (last dimension is 9)
      [Returns]
          6D rotation representation (last dimension is 6)
    """
    tensor_shape = mat.get_shape().as_list()

    if not ((tensor_shape[-1] == 3 and tensor_shape[-2] == 3) or (tensor_shape[-1] == 9)):
      raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
          but found tensor with shape {}".format(tensor_shape[-1]))
    mat = tf.reshape(mat, [-1, 3, 3])
    r6d = tf.concat([mat[...,0], mat[...,1]], axis=-1)
    if len(tensor_shape) == 1:
      r6d = tf.reshape(r6d, [6])
    return r6d

  @staticmethod
  # given rotation matrices (shape: tracks x frames x joints x 3 x 3)
  # return tf tensor ort6d representation (shape: tracks x frames x joints x 6)
  def rots_to_ort6d(rots):
    return tf.reshape(Tools.tf_matrix_to_rotation6d(tf.convert_to_tensor(rots)), [rots.shape[0],rots.shape[1],rots.shape[2],6])

  @staticmethod
  # given tf tensor ort6d representation (shape: tracks x frames x joints x 6)
  # return np array rotation matrices (shape: tracks x frames x joints x 3 x 3)
  def ort6d_to_rots(ort6d, tracks, frames, joints):
    return np.array(tf.reshape(Tools.tf_rotation6d_to_matrix(ort6d), [tracks,frames,joints,3,3]))

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

  @staticmethod
  def draw_confusion_matrix(cm, names):
    names = names.flatten()
    ax = sns.heatmap(np.array(cm) / 10, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(names)
    ax.yaxis.set_ticklabels(names)
    ## Display the visualization of the Confusion Matrix.
    plt.show()

class Metrics(object):
  @staticmethod
  def confusion_matrix(critic, n_classes, n_samples, dataset, smoothen=True, val=False):
    cm = np.zeros((n_classes,n_classes))
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
      for label in range(n_classes):
        for i in range(n_samples):
          [labels_real, X_real], y_real = dataset.generate_real_samples(1, smoothen, val=val)
          while int(labels_real.numpy().item()) != label:
            [labels_real, X_real], y_real = dataset.generate_real_samples(1, smoothen, val=val)
          with tf.device('/device:GPU:0'):
            probs = critic.predict([np.arange(n_classes), np.repeat(X_real, n_classes, axis=0)])
            cm[int(labels_real.numpy().item()), np.argmax(probs)] += 1
            # rows are the actual values, cols are the predicted values
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


