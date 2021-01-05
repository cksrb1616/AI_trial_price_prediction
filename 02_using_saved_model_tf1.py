import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

saver = tf.train.Saver()
model = tf.global_variables_initializer()

avg_temp = float(input('Average Temperature: '))
min_temp = float(input('Minimum Temperature: '))
max_temp = float(input('Maximum Temperature: '))
rain_fall = float(input('Probability of Rain: '))

with tf.Session() as sess:
    sess.run(model)
    save_path = "/Users/minjunchoi/Documents/GitHub/AI_trial_price_prediction/Web/model/saved.cpkt"
    saver.restore(sess, save_path)

    data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data})
    print(dict[0])