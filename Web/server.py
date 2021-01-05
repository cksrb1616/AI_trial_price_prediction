# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

app = Flask(__name__)

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러오는 객체를 선언합니다.
saver = tf.train.Saver()
model = tf.global_variables_initializer()

# 세션 객체를 생성합니다.
sess = tf.Session()
sess.run(model)

# 저장된 모델을 세션에 적용합니다.
save_path = "/Users/minjunchoi/Documents/GitHub/AI_trial_price_prediction/Web/model/saved.cpkt"
saver.restore(sess, save_path)


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template(('index.html'))
    if request.method == 'GET': # 그냥 접근했을 시 웹페이지를 보여줘라
        return render_template('index.html')
    if request.method == 'POST': # 파라미터를 받았을 시 이를 전달 받아라
        # 파라미터를 전달 받습니다.
        avg_temp = float(request.form['avg_temp'])
        min_temp = float(request.form['min_temp'])
        max_temp = float(request.form['max_temp'])
        rain_fall = float(request.form['rain_fall'])

        # 배추 가격 변수를 선언합니다.
        price = 0

        # 입력된 파라미터를 배열 형태로 준비합니다.
        data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
        arr = np.array(data, dtype=np.float32)
s
        # 입력 값을 토대로 예측 값을 찾아냅니다.
        x_data = arr[0:4]
        dict = sess.run(hypothesis, feed_dict={X: x_data})

        # 결과 배추 가격을 저장합니다.
        price = dict[0]
        return render_template('index.html', price=price) # 웹문서 형태로 답을 돌려준다


if __name__ == '__main__':
    app.run(debug=True)