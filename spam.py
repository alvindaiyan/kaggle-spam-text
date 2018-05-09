import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

"""
read data from csv file
""" 
df = pd.read_csv("spam.csv") # need to read data in utf8, have a look kaggle data cleaning challenge

print("original data:")
print(df.describe())


""" 
do in pandas way 
"""

"""
Firstly, removed columns with na
"""
spam_texts = df.dropna(axis=1)
np.random.shuffle(spam_texts.values)

print()
print("cleaned data")
print(spam_texts.describe())

spam_texts.columns = ['label', 'text']

# we shouldn't split the data set here because we need a consistent bag of word
# """
# split original dataset into train and test
# """
# train, test = train_test_split(spam_texts, test_size=0.2)
# train_label = train['label']
# train_text = train['text']
# test_label = test['label']
# test_text = test['text']

X = spam_texts['text']
label = spam_texts['label']

counter_vectorizer = CountVectorizer(decode_error='ignore') # there are some text contains invalid code
bag_of_words = counter_vectorizer.fit_transform(X.as_matrix())

"""
Print the first 10 features of the count_vec
"""
print("Feature length:\n{}".format(len(counter_vectorizer.get_feature_names())))

"""
split the data set to train and test
"""
train_X, test_X, train_label, test_label = train_test_split(bag_of_words, label, test_size=0.2)

model = AdaBoostClassifier()
model.fit(train_X, train_label)
print "AdaBoost accuracy: ", model.score(test_X, test_label)

