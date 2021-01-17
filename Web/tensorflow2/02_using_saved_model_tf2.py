import os
import tensorflow as tf
from tensorflow import keras

W = tf.Variable(tf.random.normal([4,1]), name="weight")
b = tf.Variable(tf.random.normal([1],name='bias'))

@tf.function
def mulreg(x):
    return tf.matmul(x, W) + b
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.000005)

saver = tf.compat.v1.train.Saver()
# model = tf.global_variables_initializer()

avg_temp = float(input('average temperature :'))
min_temp = float(input('minimum temperature :'))
max_temp = float(input('maximum temperature :'))
rain_fall = float(input('Probability of rain :'))

with tf.compat.v1.Session() as sess:
    # sess.run(model)
    save_path = "./minecheckpoint.cpkt"
    saver.restore(sess, save_path)

    data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:4]
    # hypothesis = mulreg(x_data)
    # cost = tf.reduce_mean(tf.square(mulreg(x_data) - y_data))
    # W_grad, b_grad = tape.gradient(cost, [W, b])
    dict = sess.run(hypothesis, feed_dict={X: x_data})
    print(dict[0])
