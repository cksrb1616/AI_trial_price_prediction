import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np


X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random.normal([4, 1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

saver = tf.compat.v1.train.Saver()
model = tf.compat.v1.global_variables_initializer()

avg_temp = float(input('Average Temperature: '))
min_temp = float(input('Minimum Temperature: '))
max_temp = float(input('Maximum Temperature: '))
rain_fall = float(input('Probability of Rain: '))

with tf.compat.v1.Session() as sess:
    sess.run(model)
    save_path = "./saved.cpkt"
    saver.restore(sess, save_path)

    data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data})
    print(dict[0])