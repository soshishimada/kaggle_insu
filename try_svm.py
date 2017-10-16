import csv
import numpy as np
import collections
from sklearn import svm
from sklearn.model_selection import train_test_split

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

def accuracy(predicted_labels,ground_truth_labels):
    subtract = np.array(map(int, ground_truth_labels)) - map(int,predicted_labels)
    print "pd: ",collections.Counter(predicted_labels)
    print "gt: ",ground_truth_labels
    return float(collections.Counter(subtract)[0]) / len(ground_truth_labels)


labels, vectors = read_csv('./kaggle/train.csv' ,1000)
# test_labels, test_vectors = read_csv('./kaggle/test.csv',10)
# one_hot_labels_test = one_hot_encode(test_labels)

""" 
divide training data and validation data
"""
N_train = int(len(labels) * 0.7)
N_validation = len(labels) - N_train

vectors_train, vectors_validation, labels_train, labels_validation = train_test_split(vectors, labels, test_size=N_validation)
# data distribution function


#create instance of SVC
clf = svm.NuSVC(kernel='rbf',nu=0.01)

#determine the hyperplane
print "determining Hyperplane..."
clf.fit(vectors_train,labels_train)
print "determined Hyperplane"
#prediction using the hyperplane
print "predicting . . . "
pd = clf.predict(vectors_validation)

#calculate the accuracy
print "accuracy: ", accuracy(pd,labels_validation)
