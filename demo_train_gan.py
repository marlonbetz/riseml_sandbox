import os
import riseml
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf


n_intermediate = 1024
dim_z = 2
height = 128
width = 128
n_features = height*width


n_samples  = tf.placeholder(dtype=tf.int32, shape=[])


z = tf.random_uniform(shape=[n_samples,dim_z],minval=-1,maxval=1)

w1_generator = tf.get_variable(
    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
    , shape=[dim_z, n_intermediate],
    name="w1_generator")

b1_generator = tf.get_variable(initializer=tf.zeros(shape=[n_intermediate]),
                               name="b1_generator")

w2_generator = tf.get_variable(
    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
    shape=[n_intermediate, n_features],
    name="w2_generator")

b2_generator = tf.get_variable(initializer=tf.zeros(shape=[n_features]),
                               name="b2_generator")

theta_gen = [w1_generator, b1_generator, w2_generator, b2_generator]

y_generator = tf.nn.sigmoid(
    tf.matmul(tf.nn.relu(tf.matmul(z, w1_generator) + b1_generator), w2_generator) + b2_generator)

X_discriminator_true = tf.placeholder(dtype=tf.float32, shape=[None, n_features], name="X_discriminator_true")


# X_discriminator_gen = tf.placeholder(dtype=tf.float32, shape=[None, n_features], name="X")

# Y_discriminator_gen = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Y")

w1_discriminator = tf.get_variable(
    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
    shape=[n_features, n_intermediate],
    name="w1_discriminator")

b1_discriminator = tf.get_variable(initializer=tf.zeros(shape=[n_intermediate]),
                                   name="b1_discriminator")

w2_discriminator = tf.get_variable(
    initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32),
    shape=[n_intermediate, 1],
    name="w2_discriminator")

b2_discriminator = tf.get_variable(initializer=tf.zeros(shape=[1]),
                                   name="b2_discriminator")

theta_disc = [w1_discriminator, b1_discriminator, w2_discriminator, b2_discriminator]

logits_discriminator_true = tf.matmul(
    tf.nn.relu(tf.matmul(X_discriminator_true, w1_discriminator) + b1_discriminator),
    w2_discriminator) + b2_discriminator

y_discriminator_true = tf.nn.sigmoid(logits_discriminator_true)

logits_discriminator_gen = tf.matmul(
    tf.nn.relu(tf.matmul(y_generator, w1_discriminator) + b1_discriminator),
    w2_discriminator) + b2_discriminator

y_discriminator_gen = tf.nn.sigmoid(logits_discriminator_gen)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_discriminator_true,
                                                                     labels=tf.ones_like(logits_discriminator_true)))

D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_discriminator_gen,
                                                                     labels=tf.zeros_like(logits_discriminator_gen)))

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_discriminator_gen,
                                                                labels=tf.ones_like(logits_discriminator_gen)))

optimizer_discriminator = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_disc)

optimizer_generator = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_gen)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())




    def predict(input_image):
        input_image = Image.open(BytesIO(input_image)).convert('RGB')

        input_image = input_image.resize((height, width), Image.ANTIALIAS)




        image = np.asarray(input_image, dtype=np.float32)

        image = image.transpose(2, 0, 1)
        image = np.mean(image,axis=0).reshape((1,-1))
        image /= 255

        sess.run(optimizer_discriminator, feed_dict={X_discriminator_true: image,
                                                     n_samples: 1})
        sess.run(optimizer_generator, feed_dict={n_samples :1})


        pred_raw = sess.run(y_generator, feed_dict={n_samples :1}) * 255
        result = np.stack((pred_raw,pred_raw,pred_raw),axis=0)



        result = result.transpose((1, 2, 0))

        med = Image.fromarray(np.uint8(result))

        output_image = BytesIO()
        med.save(output_image, format='JPEG')
        return output_image.getvalue()

    #with open("test.jpg","rb") as f:
    #    predict(f)
    riseml.serve(predict, port=os.environ.get('PORT'))
