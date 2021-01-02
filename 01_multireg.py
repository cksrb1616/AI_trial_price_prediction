import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import pandas as pd
import keras
import os

#model = tf.global_variabel_initializer()
data = read_csv('data.csv',encoding = "ISO-8859-1")
data = data.rename(columns={"³¯Â¥": "year", "Æò±Õ Ç³¼Ó": "avgTemp", "ÃÖÀú ±â¿Â": "minTemp","ÃÖ°í ±â¿Â":"maxTemp","°­¼ö·®":"rainFall","Æò±Õ °¡°Ý":"avgPrice"}, errors="raise")

def clean(x):
    x = x.replace("-", "")
    return int(x)
data['year'] = data['year'].apply(clean)

xy = np.array(data, dtype=np.float32)

# H(x1,x2,x3,x4) = x1w1+x2w2+x3w3+x4w4
# this can be H(X) = XW

x_data = xy[:,1:-1]
y_data = xy[:,[-1]]

# X = tf.placeholder(tf.float32, shape=[None,4])
# Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random.normal([4,1]), name="weight")
b = tf.Variable(tf.random.normal([1],name='bias'))

@tf.function
def mulreg(x):
    return tf.matmul(x, W) + b
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.000005)
# train = optimizer.minimize(cost)

# hypothesis = tf.matmul(X,W)+b
# cost = tf.reduce_mean(tf.square(hypothesis-Y))
# optimizer = tf.train.GradientDescentoptimizer(learning_rate=0.000005)
# train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

#Training
for step in range(100001):
    with tf.GradientTape() as tape:
        hypothesis = mulreg(x_data)
        cost = tf.reduce_mean(tf.square(mulreg(x_data) - y_data))
        W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(0.000005 * W_grad)
    b.assign_sub(0.000005 * b_grad)

    if step%500 ==0:
        print("#",step,"손실 비용:",cost.numpy())
        print("-배추가격:",hypothesis.numpy()[0])
#        print("#%s \t cost: %s" % (step, cost.numpy()))

# for step in range(100001):
#     cost_,hypo_ = sess.run([cost,hypothesis,train], feed_dict={X:x_data,Y:y_data})
#     if step%500 ==0:
#         print("#",step,"손실 비용:",cost_)
#         print("-배추가격:",hypo_[0])
saver = tf.compat.v1.train.Saver([W,b])
sess = tf.compat.v1.Session()
save_path = saver.save(sess,"/Users/minjunchoi/Documents/GitHub/AI_trial_price_prediction/minecheckpoint.cpkt")
print("model is saved")