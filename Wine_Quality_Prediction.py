# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 10:49:54 2019

@author: Sridhar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
#import os
#print(os.listdir("../input"))

df = pd.read_csv('winequality-red.csv', sep=';')
df.info()

#For White Wine Quality
df = pd.read_csv('winequality-white.csv', sep=';')
df.info()
##Data Visualization##

#Correlation based on Quality

df['grade'] = 1 # good
df.grade[df.quality < 7] = 0 # not good

plt.figure(figsize = (8,8))
labels = df.grade.value_counts().index
plt.pie(df.grade.value_counts(), autopct='%1.1f%%')
plt.legend(labels, loc="best")
plt.axis('equal')
plt.title('Quality Pie Chart')
plt.show()
print('The good quality wines count for ',round(df.grade.value_counts(normalize=True)[1]*100,1),'%.')

#Pairplot
sns.pairplot(df, hue='grade')
plt.show()


mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


plt.subplots(figsize = (12,12))
sns.heatmap(df.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
plt.show()

#Data Prediction
#The several models are used for the prediction.

#1. Decision Tree
#2. Random Forest
#3. KNeighbors
#4. GaussianNB
#5. SVC
#6. XGBoost

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost
from sklearn.metrics import accuracy_score

df_train_features = df.drop(['quality','grade'], axis =1)
n = 11

x_train, x_test, y_train, y_test = train_test_split(df_train_features, df['grade'], test_size=0.1, random_state=7)

x_train_mat = x_train.values.reshape((len(x_train), n))
x_test_mat = x_test.values.reshape((len(x_test), n))

##############################################################################
# Create Predictive Models
##############################################################################
print('Start Predicting...')

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train_mat,y_train)
tree_pred = decision_tree.predict(x_test_mat)

rf = RandomForestClassifier()
rf.fit(x_train_mat,y_train)
rf_pred = rf.predict(x_test_mat)

KN = KNeighborsClassifier()
KN.fit(x_train_mat,y_train)
KN_pred = KN.predict(x_test_mat)

Gaussian = GaussianNB()
Gaussian.fit(x_train_mat,y_train)
Gaussian_pred = Gaussian.predict(x_test_mat)

svc = SVC()
svc.fit(x_train_mat,y_train)
svc_pred = svc.predict(x_test_mat)

xgb = xgboost.XGBClassifier()
xgb.fit(x_train_mat,y_train)
xgb_pred = xgb.predict(x_test_mat)

print('...Complete')

##############################################################################
# Obtain Accuracy Scores for the test set
##############################################################################
print('Decision Tree:', accuracy_score(y_test, tree_pred)*100,'%')
print('Random Forest:', accuracy_score(y_test, rf_pred)*100,'%')
print('KNeighbors:',accuracy_score(y_test, KN_pred)*100,'%')
print('GaussianNB:',accuracy_score(y_test, Gaussian_pred)*100,'%')
print('SVC:',accuracy_score(y_test, svc_pred)*100,'%')
print('XGB:',accuracy_score(y_test, xgb_pred)*100,'%')

##############################################################################
# Obtain Accuracy Scores
# Each classifier has a different random state.
##############################################################################
k = [10,20,30,40,50]
for i in k:
    rf_tune = RandomForestClassifier(n_estimators=50, random_state=i)
    rf_tune.fit(x_train_mat,y_train)
    y_pred = rf_tune.predict(x_test_mat)
    print(accuracy_score(y_test, y_pred)*100,'%')
    
##############################################################################
# Input all train data
##############################################################################
x_train_check = df_train_features.values.reshape((len(df_train_features), n))
x_test_check = df['grade'].values.reshape((len(df['grade']), 1))

k = [10,20,30,40,50]
for i in k:
    rf_tune = RandomForestClassifier(n_estimators=50, random_state=i)
    rf_tune.fit(x_train_mat,y_train)
    yy_pred = rf_tune.predict(x_train_check)
    print(accuracy_score(x_test_check, yy_pred)*100,'%')
    
plt.figure(figsize = (20,8))
domain = np.linspace(1,100,len(y_pred)) 
plt.plot(domain, rf_pred,'o')
plt.plot(domain, y_test,'o')
plt.legend(('Prediction','Actual value'))
plt.show()
# =============================================================================
# 
# =============================================================================



#GridsearchCv mostly works in SVM. to finding the best estimator
from sklearn.model_selection import GridSearchCV

param_grid={'bootstrap':[True],'n_estimators':[10,20,50,100]}
classifier_grid= RandomForestClassifier()
grid_search=GridSearchCV(classifier_grid, param_grid, cv=10, n_jobs=-1)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_estimator_

#Random forest algorithms
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
accuracy=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

#Cross validation method using k-fold.To improve accuracy
from sklearn.model_selection import cross_val_score
cv_acc=cross_val_score(classifier, x_train, y_train, cv=10)

cv_acc.mean()

cv_acc.std()

#Creating a Decision Tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

feature_cols=df_train_features.columns

dot_data = StringIO()
export_graphviz(decision_tree, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Decision_Tree.png')
Image(graph.create_png())