#!/usr/bin/env python
# coding: utf-8

# In[68]:


import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler
import time


# In[69]:


data = pd.read_csv("evaluations.csv")


# In[70]:



plot_cols = data.columns[4:]


target = data["winner"] == "White"
# features = data[list(data.columns)[4:]]
features = data[plot_cols]

scaler = StandardScaler()

#scaling features
features_transformed = scaler.fit_transform(features)
features_transformed = pd.DataFrame(features_transformed, columns=features.columns)

x_train, x_test, y_train, y_test = train_test_split(features_transformed, target, test_size=0.25, random_state=0)


# In[71]:


# In[72]:


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
# estimator = DecisionTreeClassifier(max_depth=5)
estimator = LogisticRegression(solver="lbfgs")
selector = RFE(estimator, n_features_to_select=20, step=1)
selector = selector.fit(x_train, y_train)
selector.support_
new_features = []
support = selector.support_
for i in range(len(features_transformed.columns)):
    if support[i]:
        new_features.append(features_transformed.columns[i])


features_transformed = features_transformed[new_features]
x_train, x_test, y_train, y_test = train_test_split(features_transformed, target, test_size=0.25, random_state=0)


# In[73]:
from sklearn.model_selection import GridSearchCV

#parameters we will tune
estimator_params = {"LogReg":{
                               'C':[0.001, 0.01,0.025, 0.1, 1, 2, 10, 100]}, 
              "Nearest Neighbors": {'n_neighbors': [3, 5, 7, 10]},
              "Linear SVM": {'kernel':["linear"],
                             'C':[0.001, 0.01, 0.025, 0.1, 1, 2, 10, 100], },
            "RBF SVM": {'gamma':[0.1,0.5, 1,2,4,8], 
                        'C':[0.001, 0.01, 0.025, 0.1, 1, 2, 10, 100]},
            "Decision Tree": {'max_depth':[None, 3, 5, 10, 20], 'splitter':['best', 'random']},
                    "Random Forest": {'n_estimators':[10,100,200], 'max_depth':[None, 3, 5, 10, 20]},
                    "Neural Net": {'alpha': [ 0.0001, 0.001, 0.01, .01],
                                  'hidden_layer_sizes': [(8,8,4),(8,8,8), (10,), (8,), (10,8), (10,8,4)]}
                   }
names = list(estimator_params.keys())
classifiers = [
		LogisticRegression(), KNeighborsClassifier(), SVC(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier()]
def grid_search():
    for i in range(len(names)): 
        print(names[i])
        model = classifiers[i]

        #executing grid search with model
        param = estimator_params[names[i]]
        grid_search= GridSearchCV(model,param,cv=5)
        #testing performance of the grid search 
        grid_search.fit(X=x_train, y=y_train)

        print('\tBest parameters:', grid_search.best_params_)
        print('\tBest cross-validation score:', grid_search.best_score_)
        print('\tTest set score:',grid_search.score(x_test,y_test))


# In[76]:


import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grid_search()


# In[ ]:


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = DecisionTreeClassifier(max_depth=5)
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(x_train, y_train)
selector.support_


# selector.ranking_


# In[ ]:





# In[ ]:




