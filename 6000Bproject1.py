#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:56:46 2017

@author: liuyang
"""
import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import train_test_split

# KNN Classifier    
def knn_classifier(train_x, train_y):    
    from sklearn.neighbors import KNeighborsClassifier    
    model = KNeighborsClassifier()    
    model.fit(train_x, train_y)    
    return model    
# Logistic Regression Classifier    
def logistic_regression_classifier(train_x, train_y):    
    from sklearn.linear_model import LogisticRegression    
    model = LogisticRegression(penalty='l2')    
    model.fit(train_x, train_y)    
    return model     
# Random Forest Classifier    
def random_forest_classifier(train_x, train_y):    
    from sklearn.ensemble import RandomForestClassifier    
    model = RandomForestClassifier(n_estimators=8)    
    model.fit(train_x, train_y)    
    return model    
# Decision Tree Classifier    
def decision_tree_classifier(train_x, train_y):    
    from sklearn import tree    
    model = tree.DecisionTreeClassifier()    
    model.fit(train_x, train_y)    
    return model    
# SVM Classifier    
def svm_classifier(train_x, train_y):    
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    model.fit(train_x, train_y)    
    return model    
# GBDT(Gradient Boosting Decision Tree) Classifier    
def gradient_boosting_classifier(train_x, train_y):    
    from sklearn.ensemble import GradientBoostingClassifier    
    model = GradientBoostingClassifier(n_estimators=200)    
    model.fit(train_x, train_y)    
    return model    
# XGBoost Classifier
def xgb_classifier(train_x, train_y):    
    from xgboost import XGBClassifier   
    model = XGBClassifier()
    model.fit(train_x, train_y)  
    return model   

#read data
data_X = pd.read_csv('/Users/liuyang/Desktop/6000B/project1/traindata.csv', header = None)
data_y = pd.read_csv('/Users/liuyang/Desktop/6000B/project1/trainlabel.csv', header = None)
test_X = pd.read_csv('/Users/liuyang/Desktop/6000B/project1/testdata.csv', header = None)

#preprocessing the data
X_mean = data_X.mean(axis = 0)
X_std = data_X.std(axis = 0)
train = (data_X - X_mean)/X_std
test = (test_X - X_mean)/X_std
train['label'] = data_y
train_xy,val = train_test_split(train, test_size = 0.2,random_state=125)
tra_y = train_xy.label
tra_X = train_xy.drop(['label'],axis=1)
val_y = val.label
val_X = val.drop(['label'],axis=1)

#train the model
test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT', 'XGB']    
classifiers = {'KNN':knn_classifier,
               'LR':logistic_regression_classifier,    
               'RF':random_forest_classifier,    
               'DT':decision_tree_classifier,    
               'SVM':svm_classifier,
               'GBDT':gradient_boosting_classifier,
               'XGB': xgb_classifier}    

test_pred = []
val_pred = []
classifier_result = []

for i in xrange(len(test_classifiers)):  
    classifier = test_classifiers[i]
    eval_model = classifiers[classifier](tra_X, tra_y) 
    eval_pred = eval_model.predict(val_X)  
    val_pred.append(eval_pred)      
    accuracy = metrics.accuracy_score(val_y, eval_pred)
    classifier_result.append([classifier,accuracy])
    #predict the test data
    model = classifiers[classifier](train.drop(['label'],axis=1), train.label) 
    model_pred = model.predict(test)
    test_pred.append(model_pred)

eval_result = pd.DataFrame(val_pred).T
eval_result['pred'] = eval_result.mode(axis = 1)
accuracy = metrics.accuracy_score(val_y, eval_result['pred'])
classifier_result.append(['Ensemble',accuracy])
eval_metric = pd.DataFrame(classifier_result)
eval_metric.columns = ['classifier', 'accuracy']

test_result = pd.DataFrame(test_pred).T
test_result['pred'] = test_result.mode(axis = 1)
final_prediction = test_result['pred']
final_prediction.to_csv('/Users/liuyang/Desktop/6000B/project1/project1_20475562.csv')