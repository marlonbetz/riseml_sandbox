import os
import riseml
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf


height = 128
width = 128
dim_hidden = 2
X  = tf.placeholder(dtype=tf.float32, shape=[None,height*width])


w1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[height*width, dim_hidden],
                        name="w1")

b1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden],
                        name="b1")

w2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[dim_hidden,height*width],
                        name="w2")

b2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                        shape=[height*width],
                        name="b2")

Y_logits =tf.matmul( tf.nn.relu(tf.matmul(X,w1)+b1),w2)+b2

Y_pred = tf.nn.sigmoid(Y_logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_logits,labels=X)

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
