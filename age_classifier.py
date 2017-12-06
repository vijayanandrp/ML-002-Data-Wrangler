#!/usr/bin/env python3.5
# encoding: utf-8

import random
import csv
from nltk import classify, NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier

age_file = 'age.csv'
training_percent = 0.80


def feature_extraction(_data):
    """ This function is used to extract features in a given data value"""
    # Find the digits in the given string Example - data='18-20' digits = '1820'
    digits = str(''.join(c for c in _data if c.isdigit()))
    # calculate the length of the string
    len_digits = len(digits)
    # splitting digits in to values example - digits = '1820' ages = [18, 20]
    ages = [int(digits[i:i + 2]) for i in range(0, len_digits, 2)]
    # checking for special character in the given data
    special_character = '.+-<>?'
    spl_char = ''.join([c for c in list(special_character) if c in _data])
    # handling decimal age data
    if len_digits == 3:
        spl_char = '.'
        age = "".join([str(ages[0]), '.', str(ages[1])])
        # normalizing
        age = int(float(age) - 0.5)
        ages = [age]
    # Finding the maximum, minimum, average age values
    max_age = 0
    min_age = 0
    mean_age = 0
    if len(ages):
        max_age = max(ages)
        min_age = min(ages)
    if len(ages) == 2:
        mean_age = int((max_age + min_age) / 2)
    else:
        mean_age = max_age
    # specially added for 18 years cases
    only_18 = 0
    is_y = 0
    if ages == [18]:
        only_18 = 1
        if 'y' in _data or 'Y' in _data:
            is_y = 1
    under_18 = 0
    if 1 < max_age < 18:
        under_18 = 1
    above_65 = 0
    if mean_age >= 65:
        above_65 = 1
    # verifying whether digit is found in the given string or not.
    # Example - data='18-20' digits_found=True data='????' digits_found=False
    digits_found = 1
    if len_digits == 1:
        digits_found = 1
        max_age, min_age, mean_age, only_18, is_y, above_65, under_18 = 0, 0, 0, 0, 0, 0, 0
    elif len_digits == 0:
        digits_found, max_age, min_age, mean_age, only_18, is_y, above_65, under_18 = -1, -1, -1, -1, -1, -1, -1, -1
     
    feature = {
        'ages': tuple(ages),
        'len(ages)': len(ages),
        'spl_chr': spl_char,
        'is_digit': digits_found,
        'max_age': max_age,
        'mean_age': mean_age,
        'only_18': only_18,
        'is_y': is_y,
        'above_65': above_65,
        'under_18': under_18
    }

    return feature


# Loading Data set
dataset = []
with open(age_file, newline='\n') as fp:
    input_data = csv.reader(fp, delimiter=',')
    for row in input_data:
        dataset.append((row[1:]))
data = [(actual, correction) for (actual, correction) in dataset]

# randomization of dataset
random.shuffle(data)

# feature matrix and response matrix preparation.
feature_sets = [(feature_extraction(actual), correction) for (actual, correction) in data]
cut_point = int(len(feature_sets) * training_percent)
train_set, test_set = feature_sets[:cut_point], feature_sets[cut_point:]
print(len(train_set))

random.shuffle(train_set)
nb_classifier = NaiveBayesClassifier.train(train_set)
print("Accuracy of NaiveBayesClassifier: {} ".format(classify.accuracy(nb_classifier, test_set)))
print(nb_classifier.show_most_informative_features(10))

max_classifier = MaxentClassifier.train(train_set)
print("Accuracy of MaxentClassifier: {} ".format(classify.accuracy(max_classifier, test_set)))
print(max_classifier.show_most_informative_features(10))

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

    features = feature_extraction(data)
    print(features)
    prediction = [nb_classifier.classify(features),
                  max_classifier.classify(features),
                  decision_classifier.classify(features)]

    print('NaiveBayes Classifier     : ', prediction[0])
    print('Maxent Classifier         : ', prediction[1])
    print('Decision Tree Classifier  : ', prediction[2])
    print('-'*75)
    print('(Best of 3) =              ', max(set(prediction), key=prediction.count))


