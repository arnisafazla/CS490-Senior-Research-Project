import os, sys
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
import Base_WGAN

class Base_WGAN(keras.Model):
    def __init__(
        self,
        critic,
        generator,
        config,
        dataset
    ):
        super(Base_WGAN, self).__init__()
        self.critic = critic
        self.generator = generator
        self.config = config
        self.dataset = dataset

    def compile(self, c_optimizer, g_optimizer, c_loss_fn, g_loss_fn):
        super(Base_WGAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.config.n_critic):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.config.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.critic(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.critic(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                c_cost = self.c_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                c_loss = c_cost + gp * self.config.gp_weight

            # Get the gradients w.r.t the discriminator loss
            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, config.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.critic(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"c_loss": c_loss, "g_loss": g_loss}

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
      [labels_input, z_input] = self.dataset.generate_latent_points(self.config.latent_dim, labels.shape[0], self.config.n_classes)
      outputs = self.generator.predict([labels, z_input])
      return outputs


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, n_classes=6, num_img=1, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.n_classes = n_classes

    def on_epoch_end(self, epoch, logs=None):
        # generate n_samples per label and save as images
        # prepare fake examples
        labels = np.arange(self.n_classes)
        labels = np.repeat(labels, n_samples)
        outputs = self.model.generate(labels)
        # visualize and plot poses
        position_transformed = []
        for i in range(outputs.shape[0]):
            position_transformed.append(dataset.transform(np.array([outputs[i]]))[0])
        for i, mocap_track in enumerate(position_transformed):
            fig = Tools.stickfigure(mocap_track, step=20, rows=5, title=dataset.ordinalencoder.inverse_transform([[labels[i]]]), figsize=(8,8))
            fig.savefig(os.path.join(dir, str(i) + '.png'))
            plt.close()