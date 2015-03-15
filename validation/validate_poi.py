#!/usr/bin/python


"""
    starter code for the validation mini-project
    the first step toward building your POI identifier!

    start by loading/formatting the data

    after that, it's not our code anymore--it's yours!
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



### it's all yours from here forward!
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
print "accuracy = %.3f" %acc



# sample code
'''
from sklearn.cross_validation import KFold
# number of data points and number of folds
kf = KFold(len(authors), 2) 
for train_indices, test_indices in kf:
    features_train = [word_data[ii] for ii in train_indices]
    feature_test = [word_data[ii] for ii in test_indices]
    authors_train = [authors[ii]] for ii in train_indices]
    authors_test = [authors[ii]] for ii in test_indices]

'''