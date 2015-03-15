#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification

    the list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    the actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project

    the data is stored in lists and packed away in pickle files at the end

"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        # temp_counter += 1
        if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            words = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            words = words.replace("sara", "")
            words = words.replace("shackleton", "")
            words = words.replace("sshacklensf", "") # added from lesson 11
            words = words.replace("chris", "")
            words = words.replace("germani", "")
            words = words.replace("cgermannsf", "") # added from lesson 11
            


            ### append the text to word_data
            word_data.append(words)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append(0)
            else:
                from_data.append(1)

            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

'''
print "********************"
for i in range(0, len(word_data), 2):
    if word_data[i].split(' ', 1)[0] == "tjonesnsf":
        print i, "\n"        
        print word_data[i], "\n"
'''
     
# print word_data[152]



### in Part 4, do TfIdf vectorization here
# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english')
word_data_transformed = vectorizer.fit_transform(word_data)
print len(vectorizer.get_feature_names())
print vectorizer.get_feature_names()[34597]


# code samples
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from nltk.corpus import stopwords
sw = stopwords.words("english")

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("responsiveness")
'''

