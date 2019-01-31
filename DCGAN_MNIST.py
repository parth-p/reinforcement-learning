import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

def discriminator(xdata, reuse=False):
	if (reuse):
		tf.get_variable_scope().reuse_variables()
	# Discriminator Conv. Layer variables 
	disc_W1 = tf.get_variable('disc_W1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
	disc_W2 = tf.get_variable('disc_W2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
	disc_W3 = tf.get_variable('disc_W3', [7*7*64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
	disc_W4 = tf.get_variable('disc_W4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))

	disc_B1 = tf.get_variable('disc_B1', [32], initializer=tf.constant_initializer(0))
	disc_B2 = tf.get_variable('disc_B2', [64], initializer=tf.constant_initializer(0))
	disc_B3 = tf.get_variable('disc_B3', [1024], initializer=tf.constant_initializer(0))
	disc_B4 = tf.get_variable('disc_B4', [1], initializer=tf.constant_initializer(0))

	disc1 = tf.nn.conv2d(input=xdata, filter=disc_W1, strides=[1, 1, 1, 1], padding='SAME') + disc_B1
	disc1 = tf.nn.relu(disc1)
	disc1 = tf.nn.avg_pool(disc1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	disc2 = tf.nn.conv2d(input=disc1, filter=disc_W2, strides=[1, 1, 1, 1], padding='SAME') + disc_B2
	disc2 = tf.nn.relu(disc2)
	disc2 = tf.nn.avg_pool(disc2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')	

	disc3 = tf.reshape(disc2,[-1, 7*7*64])
	disc3 = tf.matmul(disc3,disc_W3) + disc_B3
	disc3 = tf.nn.relu(disc3)

	disc4 = tf.matmul(disc3,disc_W4) + disc_B4
	# disc4 = tf.nn.relu(disc4)
	return disc4

def generator(batch_size, noise_dim):
	gen_W1 = tf.get_variable('gen_W1', [noise_dim, 7*7*64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	gen_W2 = tf.get_variable('gen_W2', [3, 3, 1, noise_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	gen_W3 = tf.get_variable('gen_W3', [3, 3, noise_dim/2, noise_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	gen_W4 = tf.get_variable('gen_W4', [1, 1, noise_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

	gen_B1 = tf.get_variable('gen_B1', [7*7*64], initializer=tf.truncated_normal_initializer(stddev=0.02))
	gen_B2 = tf.get_variable('gen_B2', [noise_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
	gen_B3 = tf.get_variable('gen_B3', [noise_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
	gen_B4 = tf.get_variable('gen_B4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))

	noise = tf.truncated_normal([batch_size, noise_dim], mean=0, stddev=1, name='noise')

	gen1 = tf.matmul(noise, gen_W1) + gen_B1
	gen1 = tf.reshape(gen1, [-1, 56, 56, 1])
	gen1 = tf.contrib.layers.batch_norm(gen1, epsilon=1e-5, scope='bn1')
	gen1 = tf.nn.relu(gen1)

	gen2 = tf.nn.conv2d(gen1, gen_W2, strides=[1, 2, 2, 1], padding='SAME') + gen_B2
	gen2 = tf.contrib.layers.batch_norm(gen2, epsilon=1e-5, scope='bn2')
	gen2 = tf.nn.relu(gen2)
	gen2 = tf.image.resize_images(gen2, [56, 56])

	gen3 = tf.nn.conv2d(gen2, gen_W3, strides=[1, 2, 2, 1], padding='SAME') + gen_B3
	gen3 = tf.contrib.layers.batch_norm(gen3, epsilon=1e-5, scope='bn3')
	gen3 = tf.nn.relu(gen3)
	gen3 = tf.image.resize_images(gen3, [56, 56])

	gen4 = tf.nn.conv2d(gen3, gen_W4, strides=[1, 2, 2, 1], padding='SAME') + gen_B4
	gen4 = tf.sigmoid(gen4)

	return gen4

sess = tf.Session()

batch_size = 50
z_dimensions = 100

x_placeholder = tf.placeholder("float", shape = [None,28,28,1], name='x_placeholder')

Gz = generator(batch_size, z_dimensions)
print(Gz)
Dx = discriminator(x_placeholder)
Dg = discriminator(Gz, reuse=True)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 0.9)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

d_loss = d_loss_fake + d_loss_real

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'disc_' in var.name]
g_vars = [var for var in tvars if 'gen_' in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
    d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)

    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

tf.summary.scalar('d_real_count', d_real_count_ph)
tf.summary.scalar('d_fake_count', d_fake_count_ph)
tf.summary.scalar('g_count', g_count_ph)

# Sanity check to see how the discriminator evaluates
# generated and real MNIST images
d_on_generated = tf.reduce_mean(discriminator(generator(batch_size, z_dimensions)))
d_on_real = tf.reduce_mean(discriminator(x_placeholder))

tf.summary.scalar('d_on_generated_eval', d_on_generated)
tf.summary.scalar('d_on_real_eval', d_on_real)

images_for_tensorboard = generator(batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 10)
merged = tf.summary.merge_all()
logdir = "tensorboard/gan/"
writer = tf.summary.FileWriter(logdir, sess.graph)
print(logdir)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

gLoss = 0
dLossFake, dLossReal = 1, 1
d_real_count, d_fake_count, g_count = 0, 0, 0
for i in range(50000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    if dLossFake > 0.6:
        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],{x_placeholder: real_image_batch})
        d_fake_count += 1

    if gLoss > 0.5:
        _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],{x_placeholder: real_image_batch})
        g_count += 1

    if dLossReal > 0.45:
        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],{x_placeholder: real_image_batch})
        d_real_count += 1

    if i % 10 == 0:
        real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
                                    d_fake_count_ph: d_fake_count, g_count_ph: g_count})
        writer.add_summary(summary, i)
        d_real_count, d_fake_count, g_count = 0, 0, 0

    if i % 1000 == 0:
        images = sess.run(generator(3, z_dimensions))
        d_result = sess.run(discriminator(x_placeholder), {x_placeholder: images})
        print("TRAINING STEP", i, "AT", datetime.datetime.now())
        for j in range(3):
            print("Discriminator classification", d_result[j])
            im = images[j, :, :, 0]
            plt.imshow(im.reshape([28, 28]), cmap='Greys')
            plt.show()

    if i % 5000 == 0:
        save_path = saver.save(sess, "models/pretrained_gan.ckpt", global_step=i)
        print("saved to %s" % save_path)