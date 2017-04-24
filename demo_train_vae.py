import os
import riseml
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf




height = 128
width = 128
dim_hidden = 1024
sigma_epsilon = 1
dim_latent = 2

X  = tf.placeholder(dtype=tf.float32, shape=[None,height*width])

epsilon = tf.random_normal(shape=[tf.shape(X)[0],dim_latent],mean=0,stddev=sigma_epsilon)



w1_encoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[height*width, dim_hidden],
                        name="w1_encoder")

b1_encoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden],
                        name="b1_encoder")

w_mu = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden,dim_latent],
                        name="w_mu")

b_mu = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_latent],
                        name="b_mu")

w_sigma = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden,dim_latent],
                        name="w_sigma")

b_sigma = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_latent],
                        name="b_sigma")

mu = tf.matmul( tf.nn.relu(tf.matmul(X,w1_encoder)+b1_encoder),w_mu)+b_mu


sigma = tf.matmul( tf.nn.relu(tf.matmul(X,w1_encoder)+b1_encoder),w_sigma)+b_sigma

kl_divergence  = 0.5 * tf.reduce_sum( 1 + tf.log(tf.pow(sigma,2)) + tf.pow(mu,2) + tf.pow(sigma,2),axis=-1)

z = mu + sigma * epsilon

#decoder

w1_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_latent,dim_hidden],
                        name="w1_decoder")

b1_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden],
                        name="b1_decoder")

w2_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden,height*width],
                        name="w2_decoder")

b2_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[height*width],
                        name="b2_decoder")


Y_logits =  tf.matmul( tf.nn.relu(tf.matmul(z,w1_decoder)+b1_decoder),w2_decoder)+b2_decoder


Y_pred = tf.nn.sigmoid(Y_logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_logits,labels=X) + kl_divergence

optimizer = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())




    def predict(input_image):
        input_image = Image.open(BytesIO(input_image)).convert('RGB')

        input_image = input_image.resize((height, width), Image.ANTIALIAS)




        image = np.asarray(input_image, dtype=np.float32)

        image = image.transpose(2, 0, 1)
        image = np.mean(image,axis=0).reshape((1,-1))
        image /= 255

        sess.run(optimizer,feed_dict={X:image})

        pred_raw = sess.run(Y_pred,feed_dict={X:image})[0].reshape((height,width)) * 255
        result = np.stack((pred_raw,pred_raw,pred_raw),axis=0)



        result = result.transpose((1, 2, 0))

        med = Image.fromarray(np.uint8(result))

        output_image = BytesIO()
        med.save(output_image, format='JPEG')
        return output_image.getvalue()

    #with open("test.jpg","rb") as f:
    #    predict(f)
    riseml.serve(predict, port=os.environ.get('PORT'))
