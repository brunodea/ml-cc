import os
import tensorflow as tf

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='w')
b = tf.Variable(tf.zeros([1]), name='b')
y_hat = w * x + b

loss = tf.reduce_mean(tf.square(y_hat - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
train = optimizer.minimize(loss, name='train')

tf.variables_initializer(tf.global_variables(), name='init')

definition = tf.Session().graph_def
directory = 'data/models'
tf.train.write_graph(definition, directory, 'model_ex1.pb', as_text=False)
