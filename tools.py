import logging
import numpy as np
import tensorflow as tf

class Tools(object):
  
  @staticmethod
  # Adapted from https://github.com/papagina/RotationContinuity/blob/master/shapenet/code/tools.py.
  # euler => tf.Tensor(batch, joints * 3) (-1, 69)
  # batch => batch_size
  # joints => no. of joints
  # rotation_matrices => tf.Tensor(batch, joints, 3, 3)
  def rotation_matrices_from_euler(euler):
    batch = euler.shape[0]
    joints = int(euler.shape[1] / 3)
    b = euler.reshape(batch, joints, 3)   
    c1 = np.cos(np.deg2rad(b[:, :, 0])).reshape(batch, joints, 1, 1)
    s1 = np.sin(np.deg2rad(b[:, :, 0])).reshape(batch, joints, 1, 1)
    c2 = np.cos(np.deg2rad(b[:, :, 2])).reshape(batch, joints, 1, 1)
    s2 = np.sin(np.deg2rad(b[:, :, 2])).reshape(batch, joints, 1, 1)
    c3 = np.cos(np.deg2rad(b[:, :, 1])).reshape(batch, joints, 1, 1)
    s3 = np.cos(np.deg2rad(b[:, :, 1])).reshape(batch, joints, 1, 1)    
    row1=np.concatenate((c2*c3,          -s2,    c2*s3         ), 3)
    row2=np.concatenate((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 3)
    row3=np.concatenate((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 3)
    matrix = np.concatenate((row1, row2, row3), 2)     
    return matrix

  # Adapted from https://github.com/papagina/RotationContinuity/blob/master/shapenet/code/tools.py.
  # rotation_matrices => tf.Tensor(batch, joints, 3, 3)
  # quaterions => (batch, joints, 3)
  @staticmethod
  def quaternions_from_rotation_matrices(matrices):
    batch=matrices.shape[0]
    joints = matrices.shape[1]
    w=torch.sqrt(1.0 + matrices[:,0,0] + matrices[:,1,1] + matrices[:,2,2]) / 2.0
    w = torch.max (w , torch.autograd.Variable(torch.zeros(batch).cuda())+1e-8) #batch
    w4 = 4.0 * w;
    x= (matrices[:,2,1] - matrices[:,1,2]) / w4 # x => (batch, joints, 1, 1)
    y= (matrices[:,0,2] - matrices[:,2,0]) / w4
    z= (matrices[:,1,0] - matrices[:,0,1]) / w4
        
    quats = torch.cat( (w.view(batch,1), x.view(batch, 1),y.view(batch, 1), z.view(batch, 1) ), 1   )
        
    return quats

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


