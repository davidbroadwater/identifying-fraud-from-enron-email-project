#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

## Discard deferred_income, deferral_payments, restricted_stock_deferred,
## director_fees, loan_advances for lack of coverage.

## Discard total_payments, total_stock_value, expenses, other, and 
## email_address for use on their own because of lack of poi NaN's 
## for training purposes. This could lead to simply the presence of 
## a value being an indicator, instead of the value itself, as desired.
## However, total_payments and total_stock_value could still be used to 
## compute fractions elsewhere. 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


### Check data coverage across classes for each feature

## Create data coverage checker helper function
def data_coverage_checker(feature_name, input_data):

	counter = 0
	poi_counter = 0
	data_coverage = []
	feature_exist_poi = 0
	feature_exist_non_poi = 0
	for person in input_data:
		counter += 1
		if input_data[person]["poi"] == True:
			poi_counter += 1 
			if input_data[person][feature_name] != "NaN":
				feature_exist_poi += 1
			else:
				continue
		
		elif input_data[person][feature_name] == "NaN":
			continue
		else:
			feature_exist_non_poi += 1

	data_coverage.append(round(float(feature_exist_poi+feature_exist_non_poi)/float(counter),2))
	data_coverage.append(round(float(feature_exist_non_poi)/float(counter - poi_counter),2))
	data_coverage.append(round(float(feature_exist_poi)/float(poi_counter),2))


	return data_coverage, poi_counter, counter


financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
	'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
	'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
	'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
	'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']


## Calculate data coverage for each feature
for feature in financial_features:
	financial_coverage, poi_count, employee_count =  data_coverage_checker(feature, data_dict)
	# print "Coverage for", feature, "\t", financial_coverage

for feature in email_features:
	email_coverage, poi_count, employee_count =  data_coverage_checker(feature, data_dict)
	# print "Coverage for", feature, "\t", email_coverage

# print "Number of Employees in Dataset:", employee_count
# print "Number of POIs:", poi_count

features_list = ['poi', 
	'total_payments', 
	'total_stock_value',
	'restricted_stock', 
	'exercised_stock_options',
	'salary', 
	'bonus',
	'long_term_incentive', 
	'deferred_income',
	'deferral_payments',
	'loan_advances',
	'to_messages', 
	'from_poi_to_this_person',
	'from_messages', 
	'from_this_person_to_poi', 
	'shared_receipt_with_poi']

import pprint

# print "Initial Features used:", pprint.pprint(features_list)

### Task 2: Remove outliers

## Remove "TOTAL" entry from dictionary
# data_dict.pop( "TOTAL", 0 )

## Investigate other max values in the dataset to make sure they
## are consistent with source information. 
import math
import numpy

data_counter=0
for feature in features_list:
	feature_values = []
	poi_count = 0
	if feature == 'poi':
		continue
	else:		
		for name in data_dict:
			if math.isnan(float(data_dict[name][feature])) == False:
				feature_values.append(float(data_dict[name][feature]))
				data_counter += 1
			else:
				feature_values.append(float(0))
				# data_counter += 1


		max_value = numpy.ndarray.max(numpy.array(feature_values))
		# print "The max value for", feature,"is:", max_value

# print data_counter
### Task 3: Create new feature(s)

## Create fractional features for: shared_receipt_with_poi, to poi, from poi,
## restricted stock/total stock value, exercised_stock_options/total stock value,
## salary/total payments, bonus/total payments, long term incentive/total payments

## Create helper function for creating new features

def computeFraction( numerator, denominator ):
    """ given some feature subset (numerator) 
        and a broader parent feature (denominator),
        return the fraction of the subset to the parent feature.
   """

    ## in case of numerator or denominator having "NaN" value, return 0.
    fraction = 0.
    import math
    if math.isnan(float(denominator)) == False and math.isnan(float(numerator)) == False:
        fraction = float(numerator) / float(denominator)
    return fraction

## Create new features

for name in data_dict:

	### Create Fractional Financial Features

    salary = data_dict[name]["salary"]
    total_payments = data_dict[name]["total_payments"]
    fractional_salary = computeFraction( salary, total_payments )
    data_dict[name]["fractional_salary"] = fractional_salary

    bonus = data_dict[name]["bonus"]
    total_payments = data_dict[name]["total_payments"]
    fractional_bonus = computeFraction( bonus, total_payments )
    data_dict[name]["fractional_bonus"] = fractional_bonus

    long_term_incentive = data_dict[name]["long_term_incentive"]
    total_payments = data_dict[name]["total_payments"]
    fractional_long_term_incentive = computeFraction( long_term_incentive, total_payments )
    data_dict[name]["fractional_long_term_incentive"] = fractional_long_term_incentive

    exercised_stock_options = data_dict[name]["exercised_stock_options"]
    total_stock_value = data_dict[name]["total_stock_value"]
    fractional_exercised_stock_options = computeFraction( exercised_stock_options, total_stock_value )
    data_dict[name]["fractional_exercised_stock_options"] = fractional_exercised_stock_options

    restricted_stock = data_dict[name]["restricted_stock"]
    total_stock_value = data_dict[name]["total_stock_value"]
    fractional_restricted_stock = computeFraction( restricted_stock, total_stock_value )
    data_dict[name]["fractional_restricted_stock"] = fractional_restricted_stock

	### Create Fractional Email Features

    from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
    to_messages = data_dict[name]["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_dict[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
    from_messages = data_dict[name]["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_dict[name]["fraction_to_poi"] = fraction_to_poi

    shared_receipt_with_poi = data_dict[name]["shared_receipt_with_poi"]
    to_messages = data_dict[name]["to_messages"]
    fraction_shared_receipt_with_poi = computeFraction( shared_receipt_with_poi, to_messages )
    data_dict[name]["fraction_shared_receipt_with_poi"] = fraction_shared_receipt_with_poi

    ### Remove features no longer needed (total_payments, total_stock_value)

    data_dict[name].pop( "total_payments", 0 )
    data_dict[name].pop( "total_stock_value", 0 )

### Update Feature List with newly created features

features_list = ['poi',
	'restricted_stock', 
	'exercised_stock_options',
	'salary', 
	'fractional_salary',
	'bonus', 
	'deferred_income',
	'deferral_payments',
	'loan_advances',
	'fractional_bonus', 
	'fractional_long_term_incentive', 
	'fractional_exercised_stock_options',
	'fractional_restricted_stock', 
	'long_term_incentive', 
	'to_messages', 
	'from_poi_to_this_person', 
	'from_messages', 
	'from_this_person_to_poi', 
	'shared_receipt_with_poi', 
	'fraction_from_poi', 
	'fraction_to_poi', 
	'fraction_shared_receipt_with_poi'
	]

# print "Features used for algorithm tuning:", pprint.pprint(features_list)

### Apply Min-Max feature scaling

from sklearn import preprocessing
import math
for feature in features_list:
	feature_values = []
	if feature == 'poi':
		continue
	else:	
		for name in data_dict:
			if math.isnan(float(data_dict[name][feature])) == False:
				### Extract feature values for scaling
				feature_values.append(float(data_dict[name][feature]))
			else:
				### Set 'NaN' values to 0 for use in sklearn
				feature_values.append(float(0))
				data_dict[name][feature] = float(0)
		### Fit scaler
		min_max_scaler = preprocessing.MinMaxScaler()
		min_max_scaler.fit(feature_values)

		for name in data_dict:
			### Apply scaler to values
   			data_dict[name][feature] = min_max_scaler.transform([float(data_dict[name][feature])])


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Use SelectKBest to determine the optimum features to use 
from sklearn.feature_selection import SelectKBest, f_classif

# selector = SelectKBest(f_classif, k=21)
# selector.fit(features, labels)
# features = selector.transform(features)
# print features.shape
# pprint.pprint(selector.scores_)
# feature_scores = zip(features_list[1:],selector.scores_)

# sorted_dict = sorted(feature_scores, key=lambda feature: feature[1], reverse = True)
# for item in sorted_dict:
#  	print item[0], item[1]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Gaussian Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# pipeline_clf = GaussianNB()

### Decision Tree
from sklearn.tree import DecisionTreeClassifier
# pipeline_clf = DecisionTreeClassifier(min_samples_split=30, criterion='entropy', random_state=42)
clf = DecisionTreeClassifier(min_samples_split=30, criterion='entropy', random_state=42)

## Determine DecisionTree Feature Importances
# clf = DecisionTreeClassifier()
# clf.fit(features, labels)
# tree_scores = zip(features_list[1:],clf.feature_importances_)
# sorted_dict = sorted(tree_scores, key=lambda feature: feature[1], reverse = True)
# for item in sorted_dict:
#  	print item[0], item[1]

## Update features_list based on DecisionTreeClassifier importances
features_list = ['poi',
	'restricted_stock', 
	'exercised_stock_options',
	'salary', 
	'fractional_salary',
	'fractional_bonus', 
	'fractional_restricted_stock', 
	'long_term_incentive', 
	'from_poi_to_this_person',  
	'shared_receipt_with_poi', 
	'fraction_from_poi', 
	'fraction_to_poi', 
	]

### AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# pipeline_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

### PCA
# from sklearn.decomposition import PCA
# pca = PCA(whiten=False)

### Support Vector Machine (SVM)
# from sklearn.svm import SVC
# pipeline_clf = SVC(C=5000, gamma=0.0001, kernel='linear', random_state=42)

### Pipeline Operators
# from sklearn.pipeline import Pipeline

## Use SelectKBest to select the best features
# anova_filter = SelectKBest(f_classif, k=15)

# clf = Pipeline([
# 	('anova', anova_filter),
# 	('pca', pca),
	# ('clf', pipeline_clf)
	# ])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


## Implement GridSearchCV to tune algorithm(s)

# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import train_test_split
# from sklearn.cross_validation import StratifiedShuffleSplit

## Create Parameter Grid

# parameters = {
	# 'anova__k': (2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21),
	# 'anova__k': [2,4,6,8,10,12,14,16,18,20],
    # 'pca__whiten': [True, False], 
    # 'pca__n_components': [6,8,10,11],			# For use with PCA
    # 'clf__min_samples_split':[2,10,20,30,40,50], 		# for use with DecisionTree
    # 'clf__criterion': ['gini','entropy']				# for use with DecisionTree
    # 'clf__n_estimators': [50,100,200],				# For use with Adaboost
    # 'clf__C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],	# for use with SVM
    # 'clf__gamma': [0.0001, 0.0005, 0.001, 0.005, 		# for use with SVM
    # 	0.01, 0.1],
    # 'clf__kernel': ['linear','rbf','poly']				# for use with SVM
# }

## Create Cross Validation object for use in GridSearchCV
# cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)

## Apply GridSearchCV to the dataset
# clf = GridSearchCV(clf, parameters, scoring = 'f1', cv=cv)
# clf.fit(features, labels)

## Set the best performing combination of parameters as the new classifier
# clf = clf.best_estimator_

## Use included tester function to assess performance using cross validation
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)