#!/usr/bin/env python3.5
# encoding: utf-8

import random
import csv
from nltk import classify, NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier

gender_file = 'gender.csv'
training_percent = 0.90


def feature_extractor(_data):
    """ This function is used to extract features in a given data value"""
    _data = _data.lower()
    f_1, f_2, f_3, f_4, l_1, l_2, l_3, l_4 = None, None, None, None, None, None, None ,None
    
    # extracting first and last 4 characters
    if len(_data) >= 4:
        f_4 = _data[:4]
        l_4 = _data[-4:]
    # extracting first and last 3 characters
    if len(_data) >= 3:
        f_3 = _data[:3]
        l_3 = _data[-3:]
    # extracting first and last 2 characters
    if len(_data) >= 2:
        f_2 = _data[:2]
        l_2 = _data[-2:]
    # extracting first and last 1 character
    if len(_data) >= 1:
        f_1 = _data[:1]
        l_1 = _data[-1:]
    
    feature = {
        'f_1': f_1,
        'f_2': f_2,
        'l_1': l_1,
        'l_2': l_2,
        'f_3': f_3,
        'f_4': f_4,
        'l_3': l_3,
        'l_4': l_4
    }

    return feature


# Loading Data set
dataset = []
with open(gender_file, newline='\n') as fp:
    input_data = csv.reader(fp, delimiter=',')
    for row in input_data:
        dataset.append((row[1:]))
data = [(actual, correction) for (actual, correction) in dataset]

# shuffle list
random.shuffle(data)
feature_sets = [(feature_extractor(source), corrected) for (source, corrected) in data]
# calculating the cut-point for data set - splitting for training and test sets
cut_point = int(len(feature_sets) * training_percent)
train_set, test_set = feature_sets[:cut_point], feature_sets[cut_point:]


nb_classifier = NaiveBayesClassifier.train(train_set)
print("Accuracy of NaiveBayesClassifier: {} ".format(classify.accuracy(nb_classifier, test_set)))
print(nb_classifier.show_most_informative_features(10))


me_classifier = MaxentClassifier.train(train_set)
print("Accuracy of MaxentClassifier: {} ".format(classify.accuracy(me_classifier, test_set)))
print(me_classifier.show_most_informative_features(10))


decision_classifier = DecisionTreeClassifier.train(train_set)
print("Accuracy of DecisionTreeClassifier: {} ".format(classify.accuracy(decision_classifier, test_set)))


print('Enter q (or) quit to end this test module')
while 1:
    data = input('\nEnter data for testing: ')
    if data.lower() == 'q' or data.lower() == 'quit':
        print('End')
        exit()

    if not len(data):
        continue

    features = feature_extractor(data)
    print(features)
    prediction = [nb_classifier.classify(features),
                  me_classifier.classify(features),
                  decision_classifier.classify(features)]

    print('NaiveBayes Classifier     : ', prediction[0])
    print('Maxent Classifier         : ', prediction[1])
    print('Decision Tree Classifier  : ', prediction[2])
    print('-'*75)
    print('(Best of 3) =              ', max(set(prediction), key=prediction.count))


