#!/usr/bin/python

# the code for this function is adapted from a quiz in lesson 11
# it's used when creating three new features related to fraction of emails
def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0
    else:
        fraction = poi_messages / float(all_messages)

    return fraction

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
# features_list = ['poi','salary']

# this feature list contains all the features available
# this list is not going to be the final features I choose
# this list is for the purpose of exploring the potential learning algorithms
features_list = ['poi','salary', 'deferral_payments',
                 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees',
                 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']
              
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
# this outlier is discoverd in lesson 7 mini-project
data_dict.pop('TOTAL', 0)

# correct a data point full of typo in the financial data
print data_dict['BELFER ROBERT']

belfer = data_dict['BELFER ROBERT']
belfer['deferral_payments'] = 'NaN'
belfer['total_payments'] = 3285
belfer['exercised_stock_options'] = 'NaN'
belfer['restricted_stock'] = 44093
belfer['restricted_stock_deferred'] = -44093
belfer['total_stock_value'] = 'NaN'
belfer['expenses'] = 3285
belfer['director_fees'] = 102500
belfer['deferred_income'] = -102500
data_dict['BELFER ROBERT'].update(belfer)

print data_dict['BELFER ROBERT']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# I create three new features and explore whether might be useful to
# predict POIs
for name in data_dict:
    data_point = data_dict[name]
    
    # "from poi to this person" as a fraction of total emails to this person 
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi
    
    # "from this person to poi" as a fraction of total emails from this person
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi
    
    # "shared receipt with poi" as a fraction of total emails to this person
    shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
    to_messages = data_point["to_messages"]
    fraction_shared_receipt_with_poi \
        = computeFraction(shared_receipt_with_poi, to_messages)
    data_point["fraction_shared_receipt_with_poi"] \
        = fraction_shared_receipt_with_poi
    
    # "exercised stock options" as a fraction of total stock value
    numerator = data_point["exercised_stock_options"]
    denominator = data_point["total_stock_value"]
    fraction_exercised_stock = computeFraction(numerator, denominator)
    data_point["fraction_exercised_stock"] = fraction_exercised_stock
    
    data_dict[name].update(data_point)

# store to my_dataset for easy export below    
my_dataset = data_dict

# add created features to features list
features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")
features_list.append("fraction_shared_receipt_with_poi")
features_list.append("fraction_exercised_stock")


#####
# Task 4 and 5 are done in one block of code

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a varity of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.

# """
############################################################################
# Approach 1
# manually select features by intuition and trial and error
# manually try and tune classifiers
#
# I choose to report the result of decision tree classifer
#
# important:
# when using method 2 (see below), comment out the whole block of approach 1 
############################################################################

# step 1, select features; plot features; and report some statistics

# note that this features_list variable overwrites the previous one 
features_list = ['poi', 'exercised_stock_options']
# features_list = ['poi', 'fraction_exercised_stock'] 
# features_list = ['poi', 'exercised_stock_options', 'fraction_exercised_stock']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# 1) plot the one feature selected
# 2) count the number of data points after forming labels and features
# Note that data points with 'exercised_stock_options' == NaN are removed
# because 'exercised_stock_options' is the only feature selected here

import matplotlib.pyplot as plt
poi_count = 0
for name in data_dict:
    data_point = data_dict[name]
    if data_point['poi'] == 1:
        y = 1        
        marker = 'ro'
        poi_count = poi_count + 1
    else:
        y = 0        
        marker = 'g*'
     
    x = data_point['exercised_stock_options']
    # x = data_point['fraction_exercised_stock']
    plt.plot(x, y, marker, alpha = 0.5)
# used for another plot that limits the range of x-axis
# plt.xlim(xmin = 0, xmax = 10000000)

plt.ylim(ymin = -1, ymax = 2)
plt.xlabel('Exercised Stock Options (all data)')

# for other graphs
# plt.xlabel('Exercised Stock Options (Max = 10000000)')
# plt.xlabel('Fraction Exercised Stock Options')

plt.ylabel('POI Indicator (0 - Non-POI; 1 - POI)')
plt.show()

print "Number of total data points: ", len(data_dict)
print "Number of total POIs: ", sum(x['poi'] for x in data_dict.values())
print "Number of total data points (after feature selection): ", len(features)
print "Number of total POIs (after feature selection): ", poi_count  


# step 2) try and tune classifers
'''
# Naive Bayes
# meet the requirement if chosing features below ('poi' is the label)
# ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
'''


# Decision Tree
# meet the requirement if chosing features below ('poi' is the label)
# ['poi', 'exercised_stock_options']
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split = 80)


'''
# check why decision tree would decison tree is doing so well
# fit to full data and print out the diction tree to a pdf file
# print out the decision tree graph to a pdf file
clf.fit(features, labels)
from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO()
from sklearn import tree
tree.export_graphviz(clf, out_file = dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("dt_graph.pdf") 
'''

'''
# Random Forest
# Similar result as decision tree
# meet the requirement if chosing features below ('poi' is the label)
# ['poi', 'exercised_stock_options']
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(min_samples_split = 35)
'''

######### Approach 1 end
# """



""" 
############################################################################
# Approach 2
# use feature selection algorithm
# pipeline the feature selection, SelectKBest(), and a classifier
# manually try and tune classifiers
#
# I choose to report the result of naive Bayes classifer
#
# important:
# when using method 1 (see above), comment out the whole block of approach 2 
#############################################################################

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Naive Bayes
# meet the requirement
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB

chain = [('feature_selection', SelectKBest(f_classif, k = 5)),
              ('naive_bayes', GaussianNB())]

clf = Pipeline(chain)


'''
# SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC

chain = [('feature_selection', SelectKBest(f_classif, k = 5)),
              ('support_vector_classification', SVC(kernel="rbf", C = 1000))]

clf = Pipeline(chain)
'''

'''
# Decision Tree
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier

chain = [('feature_selection', SelectKBest(f_classif, k = 5)),
             ('Decision_Tree',
              DecisionTreeClassifier(min_samples_split = 80))]

clf = Pipeline(chain)
'''

'''
# Random forest
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

chain = [('feature_selection', SelectKBest(f_classif, k = 5)),
             ('Random_Forest', RandomForestClassifier(min_samples_split = 30))]

clf = Pipeline(chain)
'''

'''
# AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import AdaBoostClassifier

chain = [('feature_selection', SelectKBest(f_classif, k = 8)),
             ('AdaBoost', AdaBoostClassifier(n_estimators = 20))]

clf = Pipeline(chain)
'''

# my tester to rank the features selected
print "\n********************************************************"
from my_tester import my_test_classifier
my_test_classifier(clf, my_dataset, features_list)
print "********************************************************\n"

########## Approach 2 ends
"""


### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)





# abandon the code block
# keep it here for potential other use in the future
'''
# SelectKBest applies to the full data set
# Abandoned as it's not a good practice
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selector = SelectKBest(f_classif, k = 4)
features = selector.fit_transform(features, labels)

# find the features that are selected (true or false)
idx = selector.get_support().tolist()

# 'poi' is the first in the features_list and should always be there
idx = [True] + idx
features_list = [d for (d, keep) in zip(features_list, idx) if keep]

print "selected feature list", features_list

# double check
print features[0]
print data_dict["ALLEN PHILLIP K"]
'''

