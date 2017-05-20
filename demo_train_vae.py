import os
import riseml
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf




height = 256
width = 256
n_colors = 3
dim_hidden = 1024
log_var_epsilon = 1
dim_latent = 2
beta = 10

X  = tf.placeholder(dtype=tf.float32, shape=[None,height*width*n_colors])

epsilon = tf.random_normal(shape=[tf.shape(X)[0],dim_latent],mean=0,stddev=log_var_epsilon)



w1_encoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[height*width*n_colors, dim_hidden],
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

w_log_var = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden,dim_latent],
                        name="w_log_var")

b_log_var = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_latent],
                        name="b_log_var")

mu = tf.matmul( tf.nn.relu(tf.matmul(X,w1_encoder)+b1_encoder),w_mu)+b_mu


log_var = tf.matmul( tf.nn.relu(tf.matmul(X,w1_encoder)+b1_encoder),w_log_var)+b_log_var

kl_divergence  = 0.5 * tf.reduce_sum( 1 +log_var - tf.pow(mu,2) - tf.exp(log_var),axis=-1)

z = mu + tf.exp(log_var/2) * epsilon

#decoder

w1_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_latent,dim_hidden],
                        name="w1_decoder")

b1_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden],
                        name="b1_decoder")

w2_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden,height*width*n_colors],
                        name="w2_decoder")

b2_decoder = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[height*width*n_colors],
                        name="b2_decoder")


Y_logits =  tf.matmul( tf.nn.relu(tf.matmul(z,w1_decoder)+b1_decoder),w2_decoder)+b2_decoder


Y_pred = tf.nn.sigmoid(Y_logits)

loss = tf.reduce_mean(
    tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_logits, labels=X), axis=-1) + beta *  kl_divergence)

optimizer = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())




    def predict(input_image):
        input_image = Image.open(BytesIO(input_image)).convert('RGB')

        input_image = input_image.resize((height, width), Image.ANTIALIAS)




        image = np.asarray(input_image, dtype=np.float32)

        image = image.transpose(2, 0, 1).reshape((1,-1))
        #image = np.mean(image,axis=0).reshape((1,-1))
        image /= 255

        sess.run(optimizer,feed_dict={X:image})

        pred_raw = sess.run(Y_pred,feed_dict={X:image})[0].reshape((n_colors,height,width)) * 255
        result = pred_raw
        #result = np.stack((pred_raw,pred_raw,pred_raw),axis=0)



        result = result.transpose((1, 2, 0))

        med = Image.fromarray(np.uint8(result))

        output_image = BytesIO()
        med.save(output_image, format='JPEG')
        return output_image.getvalue()

    #with open("test.jpg","rb") as f:
    #    predict(f)
    riseml.serve(predict, port=os.environ.get('PORT'))
