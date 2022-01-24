import os, sys, logging, json
import matplotlib.pyplot as plt
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
os.chdir(main_dir)
from tools import Tools, Metrics

class Base_WGAN(keras.Model):
    def __init__(
        self,
        dest_dir,
        name,
        model_load,     # None or path to a specific epoch
        critic,
        generator,
        config,
        dataset
    ):
        super(Base_WGAN, self).__init__()
        self.dataset = dataset
        if model_load == None:
          self.critic = critic
          self.generator = generator
          self.name = name + datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
          self.model_dir = os.path.join(dest_dir, self.name)
          self.metrics = [list() for i in range(2)]   # c_loss, g_loss
          self.start_epoch = 0
        else:
          if os.path.basename(model_load)[:5] != epoch:
            logging.error('model_load needs to be a path to an epoch folder, as in epoch_4.')
          self.critic = load_model(os.path.join(model_load, 'critic.h5'))
          self.generator = load_model(os.path.join(model_load, 'generator.h5'))
          self.model_dir = model_load[0:-len(os.path.basename(model_load))]
          with open(os.path.join(model_load, 'metrics.txt')) as file:
            self.metrics = json.load(file)
          with open(os.path.join(self.model_dir, 'config.json')) as file:
            self.config = json.load(file)
          self.start_epoch = int(re.match('.*?([0-9]+)$', model_load).group(1))
    def compile(self, c_optimizer, g_optimizer, c_loss_fn, g_loss_fn):
        super(Base_WGAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_seq, real_labels, fake_seq):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, self.config['in_shape'][0], self.config['in_shape'][1],], 0.0, 1.0)
        diff = fake_seq - real_seq
        interpolated = real_seq + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic([real_labels, interpolated], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [real_labels, interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # logdir is the general destination path for the tensorflow board logs.
    def train(self, logdir, verbose=2):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
            with open(os.path.join(self.model_dir, 'config.json'), 'w') as file:
                json.dump(self.config, file)
            plot_model(self.critic, show_shapes=True, show_layer_names=True, to_file=os.path.join(self.model_dir, 'critic.png'))
            plot_model(self.generator, show_shapes=True, show_layer_names=True, to_file=os.path.join(self.model_dir, 'generator.png'))
            device_name = tf.test.gpu_device_name()
            if device_name == '/device:GPU:0':
                with tf.device('/device:GPU:0'):
                    tensorboard = tf.keras.callbacks.TensorBoard(os.path.join(logdir, self.name), histogram_freq=1)
                    tensorboard.set_model(self.critic)
                    tensorboard.set_model(self.generator)
                    # TRAINING
                    dataset_size = self.dataset.get_size()
                    bat_per_epo = int(dataset_size / self.config['n_batch'])
                    half_batch = int(n_batch / 2)
                    if self.config['wrong_labels']:
                        other_batch = int(half_batch / 2)
                    else:
                        other_batch = half_batch
                    for epoch in range(self.start_epoch, self.config['epochs']):
                        c_loss_epoch, g_loss_epoch = list(), list()
                        for batch in range(bat_per_epo):
                            c_loss_batch = 0
                            [labels_real, X_real], y_real = self.dataset.generate_real_samples(bat_per_epo)
                            for _ in range(self.config['n_critic']):
                                # Get the latent vector
                                labels_input, z_input = Tools.generate_latent_points(self.config['latent_dim'], bat_per_epo, self.config['n_classes'])
                                with tf.GradientTape() as tape:
                                    # Generate fake images from the latent vector
                                    fake_samples = self.generator([labels_input, z_input], training=True)
                                    # Get the logits for the fake samples
                                    fake_logits = self.critic([labels_input, fake_samples], training=True)
                                    # Get the logits for the real images
                                    real_logits = self.critic([labels_real, X_real], training=True)

                                    # Calculate the discriminator loss using the fake and real image logits
                                    c_cost = self.c_loss_fn(real=real_logits, fake=fake_logits)
                                    # Calculate the gradient penalty
                                    gp = self.gradient_penalty(bat_per_epo, X_real, labels_real, fake_samples)
                                    # Add the gradient penalty to the original discriminator loss
                                    c_loss_batch += c_cost + gp * self.config['gp_weight']

                                # Get the gradients w.r.t the discriminator loss
                                c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
                                # Update the weights of the discriminator using the discriminator optimizer
                                self.c_optimizer.apply_gradients(
                                    zip(c_gradient, self.critic.trainable_variables)
                                )

                            # Train the generator
                            # Get the latent vector
                            labels_input, z_input = Tools.generate_latent_points(self.config['latent_dim'], bat_per_epo, self.config['n_classes'])
                            with tf.GradientTape() as tape:
                                # Generate fake images using the generator
                                fake_samples = self.generator([labels_input, z_input], training=True)
                                # Get the discriminator logits for fake images
                                fake_logits = self.critic([labels_input, fake_samples], training=True)
                                # Calculate the generator loss
                                g_loss_batch = self.g_loss_fn(fake_logits)

                            # Get the gradients w.r.t the generator loss
                            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
                            # Update the weights of the generator using the generator optimizer
                            self.g_optimizer.apply_gradients(
                                zip(gen_gradient, self.generator.trainable_variables)
                            )
                            self.metrics[0].append(c_loss_batch / n_critic)
                            self.metrics[1].append(g_loss_batch)
                            c_loss_epoch.append(c_loss_batch / n_critic)
                            g_loss_epoch.append(g_loss_batch)
                            if verbose == 1 or verbose == 2:
                                print('>%d, %d/%d, d_loss_fake=%.3f, d_loss_real=%.3f, d_loss_wrong=%.3f, g=%.3f' %(epoch+1, batch+1, bat_per_epo, c_loss_batch / n_critic, g_loss_batch)
                        logs = [mean(c_loss_epoch), mean(g_loss_epoch)]
                        names = ["c_loss", "g_loss"]
                        tensorboard.on_epoch_end(epoch+1, Tools.named_logs(names, logs))
                        epoch_dir = os.path.join(self.model_dir, 'epoch_' + str(epoch+1))
                        os.mkdir(epoch_dir)
                        self.generator.save(os.path.join(epoch_dir, 'generator.h5'))
                        self.critic.save(os.path.join(epoch_dir, 'critic.h5'))
                        with open(os.path.join(epoch_dir, 'metrics.txt'), 'w') as file:
                            json.dump(self.metrics, file)
                        if verbose == 2:
                            self.save_checkpoint(self.dataset, epoch_dir, n_samples=1)
                            cm = metrics.confusion_matrix(self.critic, self.config['n_classes'], self.dataset)
                            with open(os.path.join(epoch_dir, 'cm.txt'), 'w') as file:
                                json.dump(cm, file)
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

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, n_samples):
      # generate points in latent space
      labels_input, z_input = Tools.generate_latent_points(self.latent_dim, n_samples, self.n_classes)
      # predict outputs
      seq = self.generator.predict([labels_input, z_input])
      # create class labels
      y = np.ones((n_samples, 1))
      return [labels_input, tf.convert_to_tensor(seq, dtype=tf.float32)], tf.convert_to_tensor(y, dtype=tf.float32)

    def generate(self, labels):
      [labels_input, z_input] = Tools.generate_latent_points(self.config['latent_dim'], labels.shape[0], self.config['n_classes'])
      outputs = self.generator.predict([labels, z_input])
      return outputs

    # generate n_samples per label and save as images
    def save_checkpoint(self, epoch_dir, n_samples=3):
      # prepare fake examples
      labels = np.arange(self.config['n_classes'])
      labels = np.repeat(labels, n_samples)
      outputs = self.generate(labels)
      # visualize and plot poses
      position_transformed = []
      for i in range(outputs.shape[0]):
        position_transformed.append(self.dataset.transform(np.array([outputs[i]]))[0])
      for i, mocap_track in enumerate(position_transformed):
        fig = self.dataset.stickfigure(mocap_track, step=20, rows=5, title=self.dataset.ordinalencoder.inverse_transform([[labels[i]]]), figsize=(8,8))
        fig.savefig(os.path.join(epoch_dir, str(i) + '.png'))
        plt.close()