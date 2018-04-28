from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
#import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py

FLAGS = None
m_i=0
m_j=100

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  print("Mnist data is",mnist)
  print(np.shape(mnist.train.images))
  print(np.shape(mnist.train.labels))
  print(np.shape(mnist.test.images))
  print(np.shape(mnist.test.labels))

# converting training lables to 2 class intead of 10
  output=[]
  for i in range(0,len(mnist.train.labels)):
    x,=np.where(mnist.train.labels[i]==1)
    output.append(x)

  a=[]
  for j in range(0,len(output)):
    a.append(output[j]%2)

  mnist_train_labels=[]
  for k in range(0,len(a)):
    if(a[k]==1):
      mnist_train_labels.append([0,1])
    else:
      mnist_train_labels.append([1,0])

  print(np.shape(mnist_train_labels))

 # converting test lables to 2 class intead of 10 
  output1=[]
  for i in range(0,len(mnist.test.labels)):
    x1,=np.where(mnist.test.labels[i]==1)
    output1.append(x1)

  a1=[]
  for j in range(0,len(output1)):
    a1.append(output1[j]%2)

  mnist_test_labels=[]
  for k in range(0,len(a1)):
    if(a1[k]==1):
      mnist_test_labels.append([0,1])
    else:
      mnist_test_labels.append([1,0])

  print(np.shape(mnist_test_labels))

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.ones([784, 10])) #2
  #W = tf.Variable(tf.random_normal([784 , 10] , name="weights"))
  b = tf.Variable(tf.zeros([10]))#2
  y = tf.matmul(x, W) + b

  W1 = tf.Variable(tf.ones([10, 2])) # comment these #2
  #W1 = tf.Variable(tf.random_normal([10 , 2] , name="weights"))
  b1 = tf.Variable(tf.zeros([2]))
  y1=tf.matmul(y,W1) + b1

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

 #loss function
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y1)) # change to y
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  # function to get the value from train data
  def get_values():
    global m_i
    global m_j
    x_train=mnist.train.images[m_i:m_j]
    y_train=mnist_train_labels[m_i:m_j]
    m_i=(m_i+100)%55000
    m_j=(m_j+100)%55000
    return x_train,y_train

  # Train
  for _ in range(549):
    batch_xs, batch_ys = get_values()
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y1, 1), tf.argmax(y_, 1)) # for first subdivision change to y
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess_acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist_test_labels})
  sess_pred = sess.run(correct_prediction,feed_dict={x: mnist.test.images,
                                      y_: mnist_test_labels})
  # miscalssification finding
  print(sess_pred)
  print(sess_pred.shape)
  print(sess_acc)
  mis_classify=[]
  mis_output=[]
  for c_i in range(0,len(sess_pred)):
  	if(sess_pred[c_i]==False):
  		mis_classify.append(c_i)
  		mis_output.append(output1[c_i][0])
  print("Misclassify",mis_classify)
  print("Output is",mis_output)
  plt.hist(mis_output)
  plt.title("Digit Histogram")
  plt.xlabel("Value")
  plt.ylabel("Frequency")
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)