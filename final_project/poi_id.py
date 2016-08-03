#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# exploration of dataset
print "TOTAL NUMBER OF PERSONS (DATA POINTS): ", len(my_dataset)
print "TOTAL NUMBER OF FEATURES: ", len(my_dataset[my_dataset.keys()[1]])
all_poi = 0
num_missing_values = {}
for name in my_dataset:
    if my_dataset[name]['poi'] == True:
        all_poi += 1;
    for feature in my_dataset[name]:
        if my_dataset[name][feature] == "NaN":
            if not feature in num_missing_values:
                num_missing_values[feature] = 1
            else:
                num_missing_values[feature] += 1

print "TOTAL NUMBER OF PERSON OF INTEREST: ", all_poi
num_missing_values = {}
for name in my_dataset:
    for feature in my_dataset[name]:
        if my_dataset[name][feature] == "NaN":
            if not feature in num_missing_values:
                num_missing_values[feature] = 1
            else:
                num_missing_values[feature] += 1
print "COUNTS OF MISSING ENTRIES IN DIFFERENT FEATURES: ", num_missing_values

# removing outliers
# after manual inspection in data, we found that all entries for person 'LOCKHART EUGENE E' are empty(NaN), so this person could be removed as outlier
print data_dict['LOCKHART EUGENE E']
del my_dataset['LOCKHART EUGENE E']
if not 'LOCKHART EUGENE E' in my_dataset:
    print "outlier successfully removed"

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
