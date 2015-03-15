#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
# clf = clf.fit(features, labels)
pred = clf.predict(features_test)
# pred = clf.predict(features)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
# acc = accuracy_score(pred, labels)

'''
print "accuracy = %.3f" %acc
print "predict: ", pred
print "label test: ", labels_test
print "total people in test set: ", len(labels_test)
print "number of POIs in label test: ", sum(labels_test)
'''

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision = precision_score(pred, labels_test)
recall = recall_score(pred, labels_test)
print "precision: ", precision
print "recall: ", recall