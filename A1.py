# CompSci 711 - Intro to ML
# Name: Jonathan Nguyen 
# Date: 02/28/2022 
# Title: Assignment 1: Scikit Learn or sklearn
# Version: 1.0 

# Task: 
#  In this assignment you will use scikit-learn (sklearn) library to generate and evaluate decision trees and their ensembles.
#  1. Please go through sklearn.pptx slides.  If needed, go over IDLE.pptx slides.
#  2. Choose any one dataset from OpenML (https://www.openml.org/search?type=data (Links to an external site.))It must have:
#   - All numeric features (sklearn decision trees do not directly handle nominal features)
#   - Nominal target (i.e. classification task)
#   - At least 1000 examples
#  3. Build the following models and measure their AUC (if binary classification) or Accuracy (if more than two classes) through cross-validation:
#   - Decision tree with default parameters
#   - Decision tree with tuned min_samples_leaf using GridSearchCV
#   - Random Forest
#   - Bagged decision tree
#   - AdaBoosted decision tree

# Submission: 
#  1. [3 points] The Python program .py file in which you wrote your program (do not submit work saved from the Python prompt). 
#  The program should include loading the data step as well. The user should be able to run the program and generate the results by importing the file.
#  2. A short report (pdf, doc or docx file) that includes:
#   [2 points] A brief description of the dataset (what is the task, what are the features and the target)
#   [1 point] Describe the settings of the methods (any non-default parameter, what parameter values were searched for tuning, etc.)
#   [3 points] A table that shows the evaluation measures
#   [1 point] Your thoughts on the results and conclusions

# The following are packages from the libraries that are being used in the program
from sklearn import datasets 
from sklearn import tree 
from sklearn import metrics 
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Feteches dataset and loads it in variable dia 
dia = datasets.fetch_openml(data_id = 375)

# Creates a Decision Tree object called mytree
mytree = tree.DecisionTreeClassifier(criterion = "entropy")

# Trains machine learning methods using fit method with features and targets are parameters
mytree.fit(dia.data, dia.target)

# Makes precition using predict metho with the features as parameter 
predictions = mytree.predict(dia.data)

# Evaluatation Metrics on Dataset 
print("\nEvaluation Metrics:")
print("Metric Accuracy Score = ", metrics.accuracy_score(dia.target, predictions))
print("Metric F1 Score = ", metrics.f1_score(dia.target, predictions, pos_label = "tested_positive"))
print("Metric Precision Score = ", metrics.precision_score(dia.target, predictions, pos_label = "tested_positive"))
print("Metric Recall Score = ", metrics.recall_score(dia.target, predictions, pos_label = "tested_positive"))
pp = mytree.predict_proba(dia.data)
print("Metric ROC AUC Score = ", metrics.roc_auc_score(dia.target, pp[:,1]))

# Cross-Validation for Decision tree with default parameters
dtc = tree.DecisionTreeClassifier()
cv = model_selection.cross_validate(dtc, dia.data, dia.target, scoring = "accuracy" , cv = 10, return_train_score = True)
print("\nCross-Validation for Decision tree with default parameters:")
print("Test Scores: ", cv["test_score"])
print("Test Scores Mean: ", cv["test_score"].mean())
print("Training Scores Mean: ", cv["train_score"].mean())

# Cross-Validation for Decision tree with tuned min_samples_leaf using GridSearchCV
parameters = [{"min_samples_leaf":[2,4,6,8,10]}]
tuned_dtc = model_selection.GridSearchCV(dtc, parameters, scoring="accuracy", cv = 5)
cv_tuned = model_selection.cross_validate(tuned_dtc, dia.data, dia.target, scoring="accuracy", cv = 10, return_train_score = True)
print("\nCross-Validation for Decision tree with tuned min_samples_leaf using GridSearchCV")
print("Test Scores: ", cv_tuned["test_score"])
print("Test Scores Mean: ", cv_tuned["test_score"].mean())
print("Training Scores Mean: ", cv_tuned["train_score"].mean())

tuned_dtc.fit(dia.data, dia.target)
print("\nThe best parameter to set this Decision tree with tuned min_samples_leaf using GridSearchCV is ",tuned_dtc.best_params_,"")

# Cross-Validation for Random Forest
rf = RandomForestClassifier()
cv_rf = model_selection.cross_validate(rf, dia.data, dia.target, scoring = "accuracy", cv = 10, return_train_score = True)
print("\nCross-Validation for Random Forest:")
print("Test Scores: ", cv_rf ["test_score"])
print("Test Scores Mean: ", cv_rf ["test_score"].mean())
print("Training Scores Mean: ", cv_rf ["train_score"].mean())

# Cross-Validation for Bagged decision tree
bagged_dtc = BaggingClassifier()
cv_bagged = model_selection.cross_validate(bagged_dtc, dia.data, dia.target, scoring = "accuracy", cv = 10, return_train_score = True)
print("\nCross-Validation for Bagged decision tree:")
print("Test Scores: ", cv_bagged["test_score"])
print("Test Scores Mean: ", cv_bagged["test_score"].mean())
print("Training Scores Mean: ", cv_bagged["train_score"].mean())

# Cross-Validation for AdaBoosted decision tree
ada_dtc = AdaBoostClassifier()
cv_ada = model_selection.cross_validate(ada_dtc, dia.data, dia.target, scoring = "accuracy", cv = 10, return_train_score = True)
print("\nCross-Validation for AdaBoosted decision tree:")
print("Test Scores: ", cv_ada["test_score"])
print("Test Scores Mean: ", cv_ada["test_score"].mean())
print("Training Scores Mean: ", cv_ada["train_score"].mean(),"\n")
