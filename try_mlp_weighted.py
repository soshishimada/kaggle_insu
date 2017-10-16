import csv
import numpy as np
import os
import tensorflow as tf
import collections
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



def inference(X,weight):
  with tf.name_scope("fc1") as scope:
    W_fc1 = tf.Variable(tf.random_normal([57, 40], mean=0.0, stddev=0.05))
    b_fc1 = tf.Variable(tf.zeros([40]))
    h_fc1 = tf.nn.relu(tf.matmul(X, W_fc1) + b_fc1)

  with tf.name_scope('fc2') as scope:
    W_fc2 = tf.Variable(tf.random_normal([40, 2], mean=0.0, stddev=0.05))
    b_fc2 = tf.Variable(tf.zeros([2]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

  with tf.name_scope('softmax') as scope:
    y = tf.nn.softmax(h_fc2_drop)

  weighted_y = tf.multiply(y,weight)
  return weighted_y


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


with tf.Graph().as_default():
  """
  Data preparation
  """
  labels, vectors = read_csv('./kaggle/train.csv',595213)
  #test_labels, test_vectors = read_csv('./kaggle/test.csv',10)
  num_of_0 = float(collections.Counter(labels)["0"])
  num_of_1 = float(collections.Counter(labels)["1"])
  one_hot_labels = one_hot_encode(labels)
  #one_hot_labels_test = one_hot_encode(test_labels)
  ratio_0 = num_of_0/(num_of_0+num_of_1)
  ratio_1 = 1.0 - ratio_0

  """
  Model Configuation
  """

  # Variables
  x = tf.placeholder("float", [None, 57])
  y = tf.placeholder("float", [None, 2])
  keep_prob = tf.placeholder("float")
  weight = tf.constant([ratio_0,ratio_1]) #adjust the imbalanced dataset

  #inf: prediction of labels
  inf = inference(x,weight)

  loss_value=loss(y,inf)

  train_op = training(loss_value, FLAGS.learning_rate)
  acc = accuracy(inf, y)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter("./log/", sess.graph_def)
  summary_op = tf.summary.merge_all()

  """
  training
  """


 # print one_hot_labels
#  print one_hot_labels_test
#  print np.array(vectors).shape
#  print np.array(test_vectors).shape



  patience = 0
  previous_accuracy = 0.0
  for epoch in range(FLAGS.num_of_epoch):
    print "epoch: ",epoch


    train_accuracy = sess.run(acc, feed_dict={
        x: vectors,
        y: one_hot_labels,
        keep_prob: 1.0
    })
    print("train_accuracy: ", train_accuracy)

    _, summaries_str = sess.run([train_op, summary_op], feed_dict={
      x: vectors,
      y: one_hot_labels,
      keep_prob: 0.5
    })

    summary_writer.add_summary(summaries_str)

    """
    Early Stopping
    """
    if train_accuracy <= previous_accuracy:  patience += 1
    if patience >= FLAGS.patience:
      print "<early stopping>"
      break
    previous_accuracy = train_accuracy


"""
    test_accuracy = sess.run(acc, feed_dict={
          x: test_vectors,
          y: one_hot_labels_test
        })
    print("test_accuracy: ",test_accuracy)

"""


