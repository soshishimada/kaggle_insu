import csv
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections
import pandas as pd


LOG_DIR = os.path.join(os.path.dirname(__file__),"/log")
if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

flags = tf.app.flags
FLAGS = flags.FLAGS
#Learning late for gradient descend
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_of_epoch', 20, 'Number of epochs for training')
flags.DEFINE_integer('patience', 3, 'Patience for early stopping')
flags.DEFINE_integer('max_steps', 10, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size',100, 'Batch size'
                                       'Must divide evenly into the dataset sizes.')

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_tf(pred, y):
    return gini(y, pred) / gini(y, y)

def read_csv(file,num_line):
  row_vec = []
  labels = []
  with open(file) as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
      if i != 0: #skip the first row
        labels.append(row[1])
        row_vec.append(row[2:])
      if i >= num_line:
        break
      i += 1
  return labels,np.array(row_vec)

def one_hot_encode(vector):
  one_hot_vec = []
  for i in range(len(vector)):
    if vector[i] == "0":
      one_hot_vec.append(["1","0"])
    else:
      one_hot_vec.append(["0", "1"])
  return one_hot_vec

def inference(X):
  with tf.name_scope("fc1") as scope:
    W_fc1 = tf.Variable(tf.random_normal([57, 600], mean=0.0, stddev=0.05))
    b_fc1 = tf.Variable(tf.zeros([600]))
    h_fc1 = tf.nn.relu(tf.matmul(X, W_fc1) + b_fc1)

  with tf.name_scope('fc2') as scope:
    W_fc2 = tf.Variable(tf.random_normal([600, 1200], mean=0.0, stddev=0.05))
    b_fc2 = tf.Variable(tf.zeros([1200]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  with tf.name_scope('fc3') as scope:
    W_fc3 = tf.Variable(tf.random_normal([1200, 600], mean=0.0, stddev=0.05))
    b_fc3 = tf.Variable(tf.zeros([600]))
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

  with tf.name_scope('output') as scope:
    W_fco = tf.Variable(tf.random_normal([600, 2], mean=0.0, stddev=0.05))
    b_fco = tf.Variable(tf.zeros([2]))
    h_fco = tf.nn.relu(tf.matmul(h_fc3_drop, W_fco) + b_fco)

  with tf.name_scope('softmax') as scope:
    y = tf.nn.softmax(h_fco)

  return y

def loss(label,y_inf):
    # Cost Function basic term
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(label * tf.log(y_inf))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(y_inf, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_inf, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

def sad(y_inf,labels):
  diff = np.array(y_inf) - np.array(labels)
  abs = tf.abs(diff[:,1:2])
  sad = tf.reduce_sum(abs)
  return sad,y_inf[:,1:2],labels[:,1:2]


with tf.Graph().as_default():

  """
  Model Configuation
  """

  # Variables
  x = tf.placeholder("float", [None, 57])
  y = tf.placeholder("float", [None, 2])
  keep_prob = tf.placeholder("float")
  #inf: prediction of labels
  inf = inference(x)

  loss_value=loss(y,inf)

  train_op = training(loss_value, FLAGS.learning_rate)
  acc = accuracy(inf, y)
  sad = sad(inf, y)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter("./log/", sess.graph_def)
  summary_op = tf.summary.merge_all()

  """
  training
  """
  labels, vectors = read_csv('./kaggle/train.csv',595213)
  #test_labels, test_vectors = read_csv('./kaggle/test.csv',10)
  #one_hot_labels_test = one_hot_encode(test_labels)

  """ 
  divide training data and validation data
  """
  N_train = int(len(labels) * 0.7)
  N_validation = len(labels) - N_train

  vectors_train, vectors_validation, labels_train, labels_validation = train_test_split(vectors, labels,
                                                                                        test_size=N_validation)
  # data distribution function

  one_hot_labels_train = one_hot_encode(labels_train)
  one_hot_labels_test = one_hot_encode(labels_validation)
  collections.Counter(labels_train)

  max_steps = len(labels_train)/FLAGS.batch_size
  print "max steps ",max_steps
  patience = 0
  previous_accuracy = 0.0

  """
  Training with mini batch
  """
  for step in range(max_steps):
      for i in range(len(labels_train) / FLAGS.batch_size):
          # training for each batch_size
          batch = FLAGS.batch_size * i

          sess.run(train_op, feed_dict={
              x: vectors_train[batch:batch + FLAGS.batch_size],
              y: one_hot_labels_train[batch:batch + FLAGS.batch_size],
              keep_prob: 0.5
          })

      # calculate accuracy for each steps
      train_accuracy = sess.run(acc, feed_dict={
          x: vectors_train,
          y: one_hot_labels_train,
          keep_prob: 1.0
      })
      print("train_accuracy: ", train_accuracy)



      """
      Early Stopping check
      """
      if train_accuracy <= previous_accuracy:  patience += 1
      if patience >= FLAGS.patience:
          print "early stop"
          break
      previous_accuracy = train_accuracy


  test_accuracy = sess.run(acc, feed_dict={
          x: vectors_validation,
          y: one_hot_labels_test,
          keep_prob: 1.0
        })
  print("test_accuracy: ",test_accuracy)

  sad,inf,labels = sess.run(sad, feed_dict={
          x: vectors_validation,
          y: one_hot_labels_test,
          keep_prob: 1.0
        })
  #print("SAD: ",sad)
  print "Inference",inf
  print "labels",labels
  print('Gini: ', gini_tf(inf,labels))

  with open("./kaggle/tf_submission.csv", 'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(inf)