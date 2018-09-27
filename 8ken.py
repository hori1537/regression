

from __future__ import print_function


import keras
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from keras.layers.normalization import BatchNomalization
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
import keras.models

from keras.models import load_model
from keras.utils import plot_model


import xgboost as xgb
from xgboost import XGBRegressor
import xgboost
#from xgboost import plot_tree

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import sklearn.ensemble
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import _tree

#ver0.20
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

import numpy as np
import numpy
import math
import time
import pandas as pd
import pandas
import random
import os
import sys
import itertools
import h5py

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#ver0.20
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import svm

import lime
import lime.lime_tabular

#import pydot
#import graphviz

#Date from COMSOL calculated by FUJII TAKESHI

random.seed(1)

#outputs of dataset
address_ = 'D:/xgboost/'
address_ = 'C:/Users/1310202/Documents/xgboost/'
address_ = '/home/garaken/xgboost/'
address_ = 'C:/deeplearning/model/'
address_ = 'D:/for8ken/'
save_address = 'C:\deeplearning/model/sklearn/'


DATA_SIZE = 144
CSV_NAME_8ken="inputcsv_simulation.csv"


#read the dataset

df_raw_input = pandas.read_csv(open(str(address_) + str(CSV_NAME_8ken) ,encoding="utf-8_sig"))

df_features = df_raw_input.iloc[:,0:6]
features_name = df_features.columns
print(features_name)

raw_input_train, raw_input_test = train_test_split(df_raw_input, test_size=0.1)

raw_input_train = np.array(raw_input_train)
raw_input_test = np.array(raw_input_test)

#dataset
[input_train, output_train, Tmax_train,Tmin_train,Tdelta_train]=np.hsplit(raw_input_train, [6,10,11,12])

[input_test, output_test, Tmax_test,Tmin_test,Tdelta_test]=np.hsplit(raw_input_test, [6,10,11,12])

print('input ', input_test)
print('output ', output_test)

'''
input_train = [[0.3,0.5,0.6], [0.4,0.4,0.2], [0.1,0.3,0.3]]
output_train = [[0.2,9.4], [0.4,5.5], [0.9,3.5]]
input_test = [[0.3,0.1,0.6], [0.3,0.4,0.2], [0.1,0.9,0.3]]
output_test = [[0.6,4.5], [0.2,7.7], [0.6,1.1]]
'''

### the range of each input
xpot_coef = 100
xpot_min = 80 / xpot_coef
xpot_max = 100 / xpot_coef
xpot_step = 2 / xpot_coef
xpot_candidate = np.arange(start = xpot_min , stop = xpot_max+ xpot_step, step = xpot_step)
print(xpot_candidate)
print(len(xpot_candidate))

ypot_coef = 100
ypot_min = 70 / ypot_coef
ypot_max = 100 / ypot_coef
ypot_step = 2 / ypot_coef
ypot_candidate = np.arange(start = ypot_min , stop = ypot_max + ypot_step, step = xpot_step)

tend_coef = 2.5
tend_min = 2.0 / tend_coef
tend_max = 2.4 / tend_coef
tend_step = 0.1 / tend_coef
tend_candidate = np.arange(start = tend_min , stop = tend_max + tend_step, step = tend_step)

tside_coef = 1
tside_min = 0.4 / tside_coef
tside_max = 0.8 / tside_coef
tside_step = 0.1 / tside_coef
tside_candidate = np.arange(start = tside_min , stop = tside_max + tside_step, step = tside_step)

tmesh_coef = 2.5
tmesh_min = 1.7 / tmesh_coef
tmesh_max = 2.3 / tmesh_coef
tmesh_step = 0.1 / tmesh_coef
tmesh_candidate = np.arange(start = tmesh_min , stop = tmesh_max + tmesh_step, step = tmesh_step)

hter_coef = 25
hter_min = 20 / hter_coef
hter_max = 25 / hter_coef
hter_step = 1 / hter_coef
hter_candidate = np.arange(start = hter_min , stop = hter_max + hter_step, step = hter_step)

candidate_number = 1
candidate_number = candidate_number * len(xpot_candidate)
candidate_number = candidate_number * len(ypot_candidate)
candidate_number = candidate_number * len(tend_candidate)                            
candidate_number = candidate_number * len(tside_candidate)
candidate_number = candidate_number * len(tmesh_candidate)
candidate_number = candidate_number * len(hter_candidate)

candidate_number = int(candidate_number)

input_iter = list(itertools.product(xpot_candidate, ypot_candidate, tend_candidate, tside_candidate, tmesh_candidate, hter_candidate))
input_iter = np.reshape(input_iter,[candidate_number,6])
iter_input_df = pd.DataFrame(input_iter, columns = ['xpot','ypot','tend','tside','tmesh','hter'])

train_input_df = pd.DataFrame(input_train, columns = ['xpot','ypot','tend','tside','tmesh','hter'])
test_input_df = pd.DataFrame(input_test, columns = ['xpot','ypot','tend','tside','tmesh','hter'])

train_output_df = pd.DataFrame(output_train, columns = ['T1','T2','T3','T4'])
test_output_df = pd.DataFrame(output_test, columns = ['T1','T2','T3','T4'])

###########################################################################



#######  extract descision tree   ################################################

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        
        with open(str(save_address)+ 'tree.txt', 'a') as f:
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print("{}if {} <= {}:".format(indent, name, threshold))
                print("{}if {} <= {}:".format(indent, name, threshold), file=f)
                recurse(tree_.children_left[node], depth + 1)
                print("{}else:  # if {} > {}".format(indent, name, threshold), file=f)
                print("{}else:  # if {} > {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                print("{}return {}".format(indent, tree_.value[node]))
                print("{}return {}".format(indent, tree_.value[node]), file=f)

    recurse(0, 1)



##################### Linear Regression #####################
reg1_linearregression = linear_model.LinearRegression()
reg1_linearregression.fit(input_train,output_train)

print('Linear Regression mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_linearregression.predict(input_test)))

iter_linearregression_predict_df = pd.DataFrame(reg1_linearregression.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_linearregression_predict_df['Tmax-predict'] = iter_linearregression_predict_df.max(axis=1)
iter_linearregression_predict_df['Tmin-predict'] = iter_linearregression_predict_df.min(axis=1)
iter_linearregression_predict_df['Tdelta-predict'] = iter_linearregression_predict_df['Tmax-predict'] -  iter_linearregression_predict_df['Tmin-predict']
iter_linearregression_df = pd.concat([iter_input_df,iter_linearregression_predict_df], axis=1)

train_linearregression_predict_df = pd.DataFrame(reg1_linearregression.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_linearregression_predict_df['Tmax-predict'] = train_linearregression_predict_df.max(axis=1)
train_linearregression_predict_df['Tmin-predict'] = train_linearregression_predict_df.min(axis=1)
train_linearregression_predict_df['Tdelta-predict'] = train_linearregression_predict_df['Tmax-predict'] -  train_linearregression_predict_df['Tmin-predict']
train_linearregression_df = pd.concat([train_input_df, train_linearregression_predict_df, train_output_df], axis=1)

test_linearregression_predict_df = pd.DataFrame(reg1_linearregression.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_linearregression_predict_df['Tmax-predict'] = test_linearregression_predict_df.max(axis=1)
test_linearregression_predict_df['Tmin-predict'] = test_linearregression_predict_df.min(axis=1)
test_linearregression_predict_df['Tdelta-predict'] = test_linearregression_predict_df['Tmax-predict'] -  test_linearregression_predict_df['Tmin-predict']
test_linearregression_df = pd.concat([test_input_df, test_linearregression_predict_df, test_output_df], axis=1)

iter_linearregression_df.to_csv(str(save_address) + 'linearregression_iter.csv')
train_linearregression_df.to_csv(str(save_address) + 'linearregression_train.csv')
test_linearregression_df.to_csv(str(save_address) + 'linearregression_test.csv')


linearregression_intercept_df = pd.DataFrame(reg1_linearregression.intercept_)
linearregression_coef_df      = pd.DataFrame(reg1_linearregression.coef_)
linearregression_parameter    = pd.concat([linearregression_intercept_df, linearregression_coef_df])
linearregression_parameter.to_csv(str(save_address) + 'linearregression_parameter.csv')

##################### Regression of Stochastic Gradient Descent ##################### 
reg1_multiSGD = MultiOutputRegressor(linear_model.SGDRegressor(max_iter = 1000))
reg1_multiSGD.fit(input_train,output_train)

print('multiSGD mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_multiSGD.predict(input_test)))

iter_multiSGD_predict_df = pd.DataFrame(reg1_multiSGD.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_multiSGD_predict_df['Tmax-predict'] = iter_multiSGD_predict_df.max(axis=1)
iter_multiSGD_predict_df['Tmin-predict'] = iter_multiSGD_predict_df.min(axis=1)
iter_multiSGD_predict_df['Tdelta-predict'] = iter_multiSGD_predict_df['Tmax-predict'] -  iter_multiSGD_predict_df['Tmin-predict']
iter_multiSGD_df = pd.concat([iter_input_df,iter_multiSGD_predict_df], axis=1)

train_multiSGD_predict_df = pd.DataFrame(reg1_multiSGD.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_multiSGD_predict_df['Tmax-predict'] = train_multiSGD_predict_df.max(axis=1)
train_multiSGD_predict_df['Tmin-predict'] = train_multiSGD_predict_df.min(axis=1)
train_multiSGD_predict_df['Tdelta-predict'] = train_multiSGD_predict_df['Tmax-predict'] -  train_multiSGD_predict_df['Tmin-predict']
train_multiSGD_df = pd.concat([train_input_df, train_multiSGD_predict_df, train_output_df], axis=1)

test_multiSGD_predict_df = pd.DataFrame(reg1_multiSGD.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_multiSGD_predict_df['Tmax-predict'] = test_multiSGD_predict_df.max(axis=1)
test_multiSGD_predict_df['Tmin-predict'] = test_multiSGD_predict_df.min(axis=1)
test_multiSGD_predict_df['Tdelta-predict'] = test_multiSGD_predict_df['Tmax-predict'] -  test_multiSGD_predict_df['Tmin-predict']
test_multiSGD_df = pd.concat([test_input_df, test_multiSGD_predict_df, test_output_df], axis=1)

iter_multiSGD_df.to_csv(str(save_address) + 'multiSGD_iter.csv')
train_multiSGD_df.to_csv(str(save_address) + 'multiSGD_train.csv')
test_multiSGD_df.to_csv(str(save_address) + 'multiSGD_test.csv')



##################### Regression of SVR #####################
reg1_multiSVR = MultiOutputRegressor(svm.SVR(kernel='rbf', C=1))
reg1_multiSVR.fit(input_train,output_train)

print('multiSVR mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_multiSVR.predict(input_test)))

iter_multiSVR_predict_df = pd.DataFrame(reg1_multiSVR.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_multiSVR_predict_df['Tmax-predict'] = iter_multiSVR_predict_df.max(axis=1)
iter_multiSVR_predict_df['Tmin-predict'] = iter_multiSVR_predict_df.min(axis=1)
iter_multiSVR_predict_df['Tdelta-predict'] = iter_multiSVR_predict_df['Tmax-predict'] -  iter_multiSVR_predict_df['Tmin-predict']
iter_multiSVR_df = pd.concat([iter_input_df,iter_multiSVR_predict_df], axis=1)

train_multiSVR_predict_df = pd.DataFrame(reg1_multiSVR.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_multiSVR_predict_df['Tmax-predict'] = train_multiSVR_predict_df.max(axis=1)
train_multiSVR_predict_df['Tmin-predict'] = train_multiSVR_predict_df.min(axis=1)
train_multiSVR_predict_df['Tdelta-predict'] = train_multiSVR_predict_df['Tmax-predict'] -  train_multiSVR_predict_df['Tmin-predict']
train_multiSVR_df = pd.concat([train_input_df, train_multiSVR_predict_df, train_output_df], axis=1)

test_multiSVR_predict_df = pd.DataFrame(reg1_multiSVR.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_multiSVR_predict_df['Tmax-predict'] = test_multiSVR_predict_df.max(axis=1)
test_multiSVR_predict_df['Tmin-predict'] = test_multiSVR_predict_df.min(axis=1)
test_multiSVR_predict_df['Tdelta-predict'] = test_multiSVR_predict_df['Tmax-predict'] -  test_multiSVR_predict_df['Tmin-predict']
test_multiSVR_df = pd.concat([test_input_df, test_multiSVR_predict_df, test_output_df], axis=1)

iter_multiSVR_df.to_csv(str(save_address) + 'multiSVR_iter.csv')
train_multiSVR_df.to_csv(str(save_address) + 'multiSVR_train.csv')
test_multiSVR_df.to_csv(str(save_address) + 'multiSVR_test.csv')



##################### Regression of Ridge #####################
reg1_Ridge = linear_model.Ridge(alpha=1.0)
reg1_Ridge.fit(input_train, output_train)

print('Ridge mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_Ridge.predict(input_test)))

iter_ridge_predict_df = pd.DataFrame(reg1_Ridge.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_ridge_predict_df['Tmax-predict'] = iter_ridge_predict_df.max(axis=1)
iter_ridge_predict_df['Tmin-predict'] = iter_ridge_predict_df.min(axis=1)
iter_ridge_predict_df['Tdelta-predict'] = iter_ridge_predict_df['Tmax-predict'] -  iter_ridge_predict_df['Tmin-predict']
iter_ridge_df = pd.concat([iter_input_df,iter_ridge_predict_df], axis=1)

train_ridge_predict_df = pd.DataFrame(reg1_Ridge.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_ridge_predict_df['Tmax-predict'] = train_ridge_predict_df.max(axis=1)
train_ridge_predict_df['Tmin-predict'] = train_ridge_predict_df.min(axis=1)
train_ridge_predict_df['Tdelta-predict'] = train_ridge_predict_df['Tmax-predict'] -  train_ridge_predict_df['Tmin-predict']
train_ridge_df = pd.concat([train_input_df, train_ridge_predict_df, train_output_df], axis=1)

test_ridge_predict_df = pd.DataFrame(reg1_Ridge.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_ridge_predict_df['Tmax-predict'] = test_ridge_predict_df.max(axis=1)
test_ridge_predict_df['Tmin-predict'] = test_ridge_predict_df.min(axis=1)
test_ridge_predict_df['Tdelta-predict'] = test_ridge_predict_df['Tmax-predict'] -  test_ridge_predict_df['Tmin-predict']
test_ridge_df = pd.concat([test_input_df, test_ridge_predict_df, test_output_df], axis=1)

iter_ridge_df.to_csv(str(save_address) + 'ridge_iter.csv')
train_ridge_df.to_csv(str(save_address) + 'ridge_train.csv')
test_ridge_df.to_csv(str(save_address) + 'ridge_test.csv')


ridge_intercept_df = pd.DataFrame(reg1_Ridge.intercept_)
ridge_coef_df      = pd.DataFrame(reg1_Ridge.coef_)
ridge_parameter    = pd.concat([ridge_intercept_df, ridge_coef_df])
ridge_parameter.to_csv(str(save_address) + 'ridge_parameter.csv')


##################### Regression of Lasso #####################
reg1_lasso = linear_model.Lasso()
reg1_lasso.fit(input_train,output_train)

print('Lasso mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_lasso.predict(input_test)))

iter_lasso_predict_df = pd.DataFrame(reg1_lasso.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_lasso_predict_df['Tmax-predict'] = iter_lasso_predict_df.max(axis=1)
iter_lasso_predict_df['Tmin-predict'] = iter_lasso_predict_df.min(axis=1)
iter_lasso_predict_df['Tdelta-predict'] = iter_lasso_predict_df['Tmax-predict'] -  iter_lasso_predict_df['Tmin-predict']
iter_lasso_df = pd.concat([iter_input_df,iter_lasso_predict_df], axis=1)

train_lasso_predict_df = pd.DataFrame(reg1_lasso.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_lasso_predict_df['Tmax-predict'] = train_lasso_predict_df.max(axis=1)
train_lasso_predict_df['Tmin-predict'] = train_lasso_predict_df.min(axis=1)
train_lasso_predict_df['Tdelta-predict'] = train_lasso_predict_df['Tmax-predict'] -  train_lasso_predict_df['Tmin-predict']
train_lasso_df = pd.concat([train_input_df, train_lasso_predict_df, train_output_df], axis=1)

test_lasso_predict_df = pd.DataFrame(reg1_lasso.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_lasso_predict_df['Tmax-predict'] = test_lasso_predict_df.max(axis=1)
test_lasso_predict_df['Tmin-predict'] = test_lasso_predict_df.min(axis=1)
test_lasso_predict_df['Tdelta-predict'] = test_lasso_predict_df['Tmax-predict'] -  test_lasso_predict_df['Tmin-predict']
test_lasso_df = pd.concat([test_input_df, test_lasso_predict_df, test_output_df], axis=1)

iter_lasso_df.to_csv(str(save_address) + 'lasso_iter.csv')
train_lasso_df.to_csv(str(save_address) + 'lasso_train.csv')
test_lasso_df.to_csv(str(save_address) + 'lasso_test.csv')


lasso_intercept_df = pd.DataFrame(reg1_lasso.intercept_)
lasso_coef_df      = pd.DataFrame(reg1_lasso.coef_)
lasso_parameter    = pd.concat([lasso_intercept_df, lasso_coef_df])
lasso_parameter.to_csv(str(save_address) + 'lasso_parameter.csv')

##################### Regression of MultiTaskLassoCV #####################
reg1_multitasklasso = linear_model.MultiTaskLassoCV()
reg1_multitasklasso.fit(input_train,output_train)

print('MultitaskLasso mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_multitasklasso.predict(input_test)))

iter_multitasklasso_predict_df = pd.DataFrame(reg1_multitasklasso.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_multitasklasso_predict_df['Tmax-predict'] = iter_multitasklasso_predict_df.max(axis=1)
iter_multitasklasso_predict_df['Tmin-predict'] = iter_multitasklasso_predict_df.min(axis=1)
iter_multitasklasso_predict_df['Tdelta-predict'] = iter_multitasklasso_predict_df['Tmax-predict'] -  iter_multitasklasso_predict_df['Tmin-predict']
iter_multitasklasso_df = pd.concat([iter_input_df,iter_multitasklasso_predict_df], axis=1)

train_multitasklasso_predict_df = pd.DataFrame(reg1_multitasklasso.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_multitasklasso_predict_df['Tmax-predict'] = train_multitasklasso_predict_df.max(axis=1)
train_multitasklasso_predict_df['Tmin-predict'] = train_multitasklasso_predict_df.min(axis=1)
train_multitasklasso_predict_df['Tdelta-predict'] = train_multitasklasso_predict_df['Tmax-predict'] -  train_multitasklasso_predict_df['Tmin-predict']
train_multitasklasso_df = pd.concat([train_input_df, train_multitasklasso_predict_df, train_output_df], axis=1)

test_multitasklasso_predict_df = pd.DataFrame(reg1_multitasklasso.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_multitasklasso_predict_df['Tmax-predict'] = test_multitasklasso_predict_df.max(axis=1)
test_multitasklasso_predict_df['Tmin-predict'] = test_multitasklasso_predict_df.min(axis=1)
test_multitasklasso_predict_df['Tdelta-predict'] = test_multitasklasso_predict_df['Tmax-predict'] -  test_multitasklasso_predict_df['Tmin-predict']
test_multitasklasso_df = pd.concat([test_input_df, test_multitasklasso_predict_df, test_output_df], axis=1)

iter_multitasklasso_df.to_csv(str(save_address) + 'multitasklasso_iter.csv')
train_multitasklasso_df.to_csv(str(save_address) + 'multitasklasso_train.csv')
test_multitasklasso_df.to_csv(str(save_address) + 'multitasklasso_test.csv')


multitasklasso_intercept_df = pd.DataFrame(reg1_multitasklasso.intercept_)
multitasklasso_coef_df      = pd.DataFrame(reg1_multitasklasso.coef_)
multitasklasso_parameter    = pd.concat([multitasklasso_intercept_df, multitasklasso_coef_df])
multitasklasso_parameter.to_csv(str(save_address) + 'Multitasklasso_parameter.csv')



##################### Regression of Elastic Net #####################
reg1_elasticnet = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
reg1_elasticnet.fit(input_train,output_train)

print('ElasticNetCV mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_elasticnet.predict(input_test)))

iter_elasticnet_predict_df = pd.DataFrame(reg1_elasticnet.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_elasticnet_predict_df['Tmax-predict'] = iter_elasticnet_predict_df.max(axis=1)
iter_elasticnet_predict_df['Tmin-predict'] = iter_elasticnet_predict_df.min(axis=1)
iter_elasticnet_predict_df['Tdelta-predict'] = iter_elasticnet_predict_df['Tmax-predict'] -  iter_elasticnet_predict_df['Tmin-predict']
iter_elasticnet_df = pd.concat([iter_input_df,iter_elasticnet_predict_df], axis=1)

train_elasticnet_predict_df = pd.DataFrame(reg1_elasticnet.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_elasticnet_predict_df['Tmax-predict'] = train_elasticnet_predict_df.max(axis=1)
train_elasticnet_predict_df['Tmin-predict'] = train_elasticnet_predict_df.min(axis=1)
train_elasticnet_predict_df['Tdelta-predict'] = train_elasticnet_predict_df['Tmax-predict'] -  train_elasticnet_predict_df['Tmin-predict']
train_elasticnet_df = pd.concat([train_input_df, train_elasticnet_predict_df, train_output_df], axis=1)

test_elasticnet_predict_df = pd.DataFrame(reg1_elasticnet.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_elasticnet_predict_df['Tmax-predict'] = test_elasticnet_predict_df.max(axis=1)
test_elasticnet_predict_df['Tmin-predict'] = test_elasticnet_predict_df.min(axis=1)
test_elasticnet_predict_df['Tdelta-predict'] = test_elasticnet_predict_df['Tmax-predict'] -  test_elasticnet_predict_df['Tmin-predict']
test_elasticnet_df = pd.concat([test_input_df, test_elasticnet_predict_df, test_output_df], axis=1)

iter_elasticnet_df.to_csv(str(save_address) + 'elasticnet_iter.csv')
train_elasticnet_df.to_csv(str(save_address) + 'elasticnet_train.csv')
test_elasticnet_df.to_csv(str(save_address) + 'elasticnet_test.csv')


elasticnet_intercept_df = pd.DataFrame(reg1_elasticnet.intercept_)
elasticnet_coef_df      = pd.DataFrame(reg1_elasticnet.coef_)
elasticnet_parameter    = pd.concat([elasticnet_intercept_df, elasticnet_coef_df])
elasticnet_parameter.to_csv(str(save_address) + 'elasticnet_parameter.csv')


##################### Regression of Multi Task Elastic Net CV #####################
reg1_multitaskelasticnet = linear_model.MultiTaskElasticNetCV()
reg1_multitaskelasticnet.fit(input_train,output_train)

print('MultiTask ElasticNet mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_multitaskelasticnet.predict(input_test)))

iter_multitaskelasticnet_predict_df = pd.DataFrame(reg1_multitaskelasticnet.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_multitaskelasticnet_predict_df['Tmax-predict'] = iter_multitaskelasticnet_predict_df.max(axis=1)
iter_multitaskelasticnet_predict_df['Tmin-predict'] = iter_multitaskelasticnet_predict_df.min(axis=1)
iter_multitaskelasticnet_predict_df['Tdelta-predict'] = iter_multitaskelasticnet_predict_df['Tmax-predict'] -  iter_multitaskelasticnet_predict_df['Tmin-predict']
iter_multitaskelasticnet_df = pd.concat([iter_input_df,iter_multitaskelasticnet_predict_df], axis=1)

train_multitaskelasticnet_predict_df = pd.DataFrame(reg1_multitaskelasticnet.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_multitaskelasticnet_predict_df['Tmax-predict'] = train_multitaskelasticnet_predict_df.max(axis=1)
train_multitaskelasticnet_predict_df['Tmin-predict'] = train_multitaskelasticnet_predict_df.min(axis=1)
train_multitaskelasticnet_predict_df['Tdelta-predict'] = train_multitaskelasticnet_predict_df['Tmax-predict'] -  train_multitaskelasticnet_predict_df['Tmin-predict']
train_multitaskelasticnet_df = pd.concat([train_input_df, train_multitaskelasticnet_predict_df, train_output_df], axis=1)

test_multitaskelasticnet_predict_df = pd.DataFrame(reg1_multitaskelasticnet.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_multitaskelasticnet_predict_df['Tmax-predict'] = test_multitaskelasticnet_predict_df.max(axis=1)
test_multitaskelasticnet_predict_df['Tmin-predict'] = test_multitaskelasticnet_predict_df.min(axis=1)
test_multitaskelasticnet_predict_df['Tdelta-predict'] = test_multitaskelasticnet_predict_df['Tmax-predict'] -  test_multitaskelasticnet_predict_df['Tmin-predict']
test_multitaskelasticnet_df = pd.concat([test_input_df, test_multitaskelasticnet_predict_df, test_output_df], axis=1)

iter_multitaskelasticnet_df.to_csv(str(save_address) + 'multitaskelasticnet_iter.csv')
train_multitaskelasticnet_df.to_csv(str(save_address) + 'multitaskelasticnet_train.csv')
test_multitaskelasticnet_df.to_csv(str(save_address) + 'multitaskelasticnet_test.csv')


multitaskelasticnet_intercept_df = pd.DataFrame(reg1_multitaskelasticnet.intercept_)
multitaskelasticnet_coef_df      = pd.DataFrame(reg1_multitaskelasticnet.coef_)
multitaskelasticnet_parameter    = pd.concat([multitaskelasticnet_intercept_df, multitaskelasticnet_coef_df])
multitaskelasticnet_parameter.to_csv(str(save_address) + 'multitaskelasticnet_parameter.csv')

##################### Regression of DecisionTreeRegressor #####################
reg1_DTR = sklearn.tree.DecisionTreeRegressor(max_depth = 5)
reg1_DTR.fit(input_train,output_train)

print('DTR mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_DTR.predict(input_test)))

iter_DTR_predict_df = pd.DataFrame(reg1_DTR.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_DTR_predict_df['Tmax-predict'] = iter_DTR_predict_df.max(axis=1)
iter_DTR_predict_df['Tmin-predict'] = iter_DTR_predict_df.min(axis=1)
iter_DTR_predict_df['Tdelta-predict'] = iter_DTR_predict_df['Tmax-predict'] -  iter_DTR_predict_df['Tmin-predict']
iter_DTR_df = pd.concat([iter_input_df,iter_DTR_predict_df], axis=1)

train_DTR_predict_df = pd.DataFrame(reg1_DTR.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_DTR_predict_df['Tmax-predict'] = train_DTR_predict_df.max(axis=1)
train_DTR_predict_df['Tmin-predict'] = train_DTR_predict_df.min(axis=1)
train_DTR_predict_df['Tdelta-predict'] = train_DTR_predict_df['Tmax-predict'] -  train_DTR_predict_df['Tmin-predict']
train_DTR_df = pd.concat([train_input_df, train_DTR_predict_df, train_output_df], axis=1)

test_DTR_predict_df = pd.DataFrame(reg1_DTR.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_DTR_predict_df['Tmax-predict'] = test_DTR_predict_df.max(axis=1)
test_DTR_predict_df['Tmin-predict'] = test_DTR_predict_df.min(axis=1)
test_DTR_predict_df['Tdelta-predict'] = test_DTR_predict_df['Tmax-predict'] -  test_DTR_predict_df['Tmin-predict']
test_DTR_df = pd.concat([test_input_df, test_DTR_predict_df, test_output_df], axis=1)

iter_DTR_df.to_csv(str(save_address) + 'DTR_iter.csv')
train_DTR_df.to_csv(str(save_address) + 'DTR_train.csv')
test_DTR_df.to_csv(str(save_address) + 'DTR_test.csv')

tree_to_code(reg1_DTR, ['xpot','ypot','tend','tside','tmesh','hter'])



##################### Regression of RandomForestRegressor #####################
reg1_RFR = sklearn.ensemble.RandomForestRegressor()
reg1_RFR.fit(input_train,output_train)

print('RFR mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_RFR.predict(input_test)))

iter_RFR_predict_df = pd.DataFrame(reg1_RFR.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_RFR_predict_df['Tmax-predict'] = iter_RFR_predict_df.max(axis=1)
iter_RFR_predict_df['Tmin-predict'] = iter_RFR_predict_df.min(axis=1)
iter_RFR_predict_df['Tdelta-predict'] = iter_RFR_predict_df['Tmax-predict'] -  iter_RFR_predict_df['Tmin-predict']
iter_RFR_df = pd.concat([iter_input_df,iter_RFR_predict_df], axis=1)

train_RFR_predict_df = pd.DataFrame(reg1_RFR.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_RFR_predict_df['Tmax-predict'] = train_RFR_predict_df.max(axis=1)
train_RFR_predict_df['Tmin-predict'] = train_RFR_predict_df.min(axis=1)
train_RFR_predict_df['Tdelta-predict'] = train_RFR_predict_df['Tmax-predict'] -  train_RFR_predict_df['Tmin-predict']
train_RFR_df = pd.concat([train_input_df, train_RFR_predict_df, train_output_df], axis=1)

test_RFR_predict_df = pd.DataFrame(reg1_RFR.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_RFR_predict_df['Tmax-predict'] = test_RFR_predict_df.max(axis=1)
test_RFR_predict_df['Tmin-predict'] = test_RFR_predict_df.min(axis=1)
test_RFR_predict_df['Tdelta-predict'] = test_RFR_predict_df['Tmax-predict'] -  test_RFR_predict_df['Tmin-predict']
test_RFR_df = pd.concat([test_input_df, test_RFR_predict_df, test_output_df], axis=1)

iter_RFR_df.to_csv(str(save_address) + 'RFR_iter.csv')
train_RFR_df.to_csv(str(save_address) + 'RFR_train.csv')
test_RFR_df.to_csv(str(save_address) + 'RFR_test.csv')

#tree_to_code(reg1_RFR , ['xpot','ypot','tend','tside','tmesh','hter'])

##################### Regression of XGBoost #####################
reg1_multigbtree = MultiOutputRegressor(xgb.XGBRegressor())

#reg1_multigbtree_cv = GridSearchCV(reg1_multigbtree, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
#reg1_multigbtree_cv.fit(input_train,output_train)
#print('best gbregressor ', reg1_multigbtree_cv.best_params_, reg1_multigbtree_cv.best_score_)
#reg1_multigbtree= MultiOutputRegressor(xgb.XGBRegressor(**reg1_gbtree_cv.best_params_))
reg1_multigbtree.fit(input_train,output_train)

print('multigbtree mean_squared_error')
print(sklearn.metrics.mean_squared_error(output_test, reg1_multigbtree.predict(input_test)))

iter_multigbtree_predict_df = pd.DataFrame(reg1_multigbtree.predict(input_iter), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
iter_multigbtree_predict_df['Tmax-predict'] = iter_multigbtree_predict_df.max(axis=1)
iter_multigbtree_predict_df['Tmin-predict'] = iter_multigbtree_predict_df.min(axis=1)
iter_multigbtree_predict_df['Tdelta-predict'] = iter_multigbtree_predict_df['Tmax-predict'] -  iter_multigbtree_predict_df['Tmin-predict']
iter_multigbtree_df = pd.concat([iter_input_df,iter_multigbtree_predict_df], axis=1)

train_multigbtree_predict_df = pd.DataFrame(reg1_multigbtree.predict(input_train), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
train_multigbtree_predict_df['Tmax-predict'] = train_multigbtree_predict_df.max(axis=1)
train_multigbtree_predict_df['Tmin-predict'] = train_multigbtree_predict_df.min(axis=1)
train_multigbtree_predict_df['Tdelta-predict'] = train_multigbtree_predict_df['Tmax-predict'] -  train_multigbtree_predict_df['Tmin-predict']
train_multigbtree_df = pd.concat([train_input_df, train_multigbtree_predict_df, train_output_df], axis=1)

test_multigbtree_predict_df = pd.DataFrame(reg1_multigbtree.predict(input_test), columns = ['T1-predict','T2-predict','T3-predict','T4-predict'])
test_multigbtree_predict_df['Tmax-predict'] = test_multigbtree_predict_df.max(axis=1)
test_multigbtree_predict_df['Tmin-predict'] = test_multigbtree_predict_df.min(axis=1)
test_multigbtree_predict_df['Tdelta-predict'] = test_multigbtree_predict_df['Tmax-predict'] -  test_multigbtree_predict_df['Tmin-predict']
test_multigbtree_df = pd.concat([test_input_df, test_multigbtree_predict_df, test_output_df], axis=1)

iter_multigbtree_df.to_csv(str(save_address) + 'multigbtree_iter.csv')
train_multigbtree_df.to_csv(str(save_address) + 'multigbtree_train.csv')
test_multigbtree_df.to_csv(str(save_address) + 'multigbtree_test.csv')


'''
import matplotlib.pyplot as plt

importances = pd.Series(reg1_multigbtree.feature_importances_)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()
'''


sys.exit()

##################### Save as CSV file #####################

'''
##################### LIME Explainer #####################
#explainer1 = lime.lime_tabular.LimeTabularExplainer(output_train, feature_names=feature_names, kernel_width=3)

explainer1 = lime.lime_tabular.LimeTabularExplainer(input_train, feature_names=feature_names, class_names=['haze'], verbose=True, mode='regression')

np.random.seed(1)
i = 3
#exp = explainer.explain_instance(test[2], predict_fn, num_features=10)
exp = explainer.explain_instance(test[i], reg1_SVR.predict, num_features=5)

sys.exit()
# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path= str(address_) + 'numeric_category_feat_01", show_table=True, show_all=True)

i = 3
exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path=str(address_) + 'numeric_category_feat_02", show_table=True, show_all=True)


# import pickle
# pickle.dump(reg, open("model.pkl", "wb"))
# reg = pickle.load(open("model.pkl", "rb"))
'''

'''
pred1_train = reg1_gbtree.predict(input_train)
pred1_test = reg1_gbtree.predict(input_test)
print(mean_squared_error(output_train, pred1_train))
print(mean_squared_error(output_test, pred1_test))

import matplotlib.pyplot as plt

importances = pd.Series(reg1_gbtree.feature_importances_)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()
'''


batch_size = 20
epochs = 500000
#epochs = 1


def get_model(num_layers, layer_size,bn_where,ac_last,keep_prob):
    model =Sequential()
    model.add(InputLayer(input_shape=(6,)))
    #model.add(Dense(layer_size))

    for i in range(num_layers):
        if num_layers != 1:

            model.add(Dense(layer_size))

            if bn_where==0 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

            model.add(Activation('relu'))

            if bn_where==1 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

            model.add(Dropout(keep_prob))

            if bn_where==2 or bn_where ==3:
                model.add(BatchNormalization(mode=0))

    if ac_last ==1:
        model.add(Activation('relu'))

    model.add(Dense(4))


    model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    return model



for patience_ in [3000]:

    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_, verbose=0, mode='auto')

    for num_layers in [4,3]:
        if num_layers !=1:
            for layer_size in[1024,512,256,128,64,32,16]:
                for bn_where in [4]:
                    for ac_last in [0]:
                        for keep_prob in [0]:

                            model =get_model(num_layers,layer_size,bn_where,ac_last,keep_prob)

                            if layer_size >= 1024:
                                batch_size = 30
                            elif num_layers >= 4:
                                batch_size = 30
                            elif bn_where ==3:
                                batch_size=30

                            else:
                                batch_size = 30


                            model.fit(input_train, output_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=0,
                                      validation_data=(input_test, output_test),
                                      callbacks=[es_cb])

                            score_test = model.evaluate(input_test, output_test, verbose=0)
                            score_train = model.evaluate(input_train, output_train, verbose=0)
                            test_predict = model.predict(input_test, batch_size=32, verbose=0)
                            train_predict = model.predict(input_train, batch_size=32, verbose=0)

                            df_test_pre = pd.DataFrame(test_predict)
                            df_train_pre = pd.DataFrame(train_predict)

                            df_test_param = pd.DataFrame(output_test)
                            df_train_param = pd.DataFrame(output_train)

                            df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                            df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                            savename = ""
                            savename = savename  + '_score_train-' + str("%.3f" % round(math.log10(score_train[0]), 3))
                            savename = savename  + '_score_test-' + str("%.3f" % round(math.log10(score_test[0]), 3))
                            savename = savename  + '_numlayer-' + str(num_layers)
                            savename = savename  + '_layersize-' + str(layer_size)
                            savename = savename  + '_bn- ' + str(bn_where)
                            savename = savename  + '_ac-' + str(ac_last)
                            savename = savename  + '_k_p-' + str(keep_prob)
                            savename = savename  + '_patience-' + str(patience_)

                            df_dens_test.to_csv('C:\deeplearning/model/csv/'
                                                + savename
                                                + '_test.csv')

                            df_dens_train.to_csv('C:\deeplearning/model/csv/'
                                                + savename
                                                + '_train.csv')

                            model.save('C:\deeplearning/model/8ken/model_8ken'
                                            + savename
                                            + '.h5')
                            #plot_model(model, to_file='C:\deeplearning/model.png')
                            #plot_model(model, to_file='model.png')


                            print('Test loss:', score_test[0])
                            print('Test accuracy:', score_test[1])


                            print('C:\deeplearning/model/8ken/model_8ken'
                                           + savename
                                           + '.h5')


                            model.summary()


                            ### evaluation of deeplearning ###
                            def eval_bydeeplearning(input):
                                output_predict = model.predict(input, batch_size = 1, verbose= 0)
                                output_predict = np.array(output_predict)

                                return output_predict

                
                            print('start the evaluation by deeplearning')
                            print('candidate is ', candidate_number)
                            
                            start_time = time.time()
                            output_iter = eval_bydeeplearning(input_iter)
                            
                            
                            iter_deeplearning_predict_df = pd.DataFrame(output_iter, columns = ['T1','T2','T3','T4'])                            
                            iter_deeplearning_predict_df['Tmax'] = iter_deeplearning_predict_df.max(axis=1)
                            iter_deeplearning_predict_df['Tmin'] = iter_deeplearning_predict_df.min(axis=1)
                            iter_deeplearning_predict_df['Tdelta'] = iter_deeplearning_predict_df['Tmax'] -  iter_deeplearning_predict_df['Tmin']

                            iter_deeplearning_df = pd.concat([iter_input_df, iter_deeplearning_predict_df], axis=1)

                            end_time = time.time()

                            total_time = end_time - start_time
                            print('total_time 1', total_time)

                            i=0
                            
                            predict_df = pd.DataFrame()
                            output_delta_temp = 100000
                            

                            '''
                            start_time = time.time()
                            print('start the for loop')
                            for xpot_, ypot_, tend_, tside_, tmesh_, hter_ in itertools.product(xpot_candidate, ypot_candidate, tend_candidate, tside_candidate, tmesh_candidate, hter_candidate):
                               
                                input_ori = [xpot_, ypot_, tend_, tside_, tmesh_, hter_]
                                
                                xpot_ = xpot_ / xpot_coef
                                ypot_ = ypot_ / ypot_coef
                                tend_ = tend_ / tend_coef
                                tside_ = tside_ / tside_coef
                                tmesh_ = tmesh_ / tmesh_coef
                                hter_ = hter_ / hter_coef
                                
                                input = [xpot_, ypot_, tend_, tside_, tmesh_, hter_]
                                
                                #print(input)
                                input = np.reshape(input, [1,6])
                                
                                output = eval_bydeeplearning(input)
                                output = output * 1000
                                output_max = float(max(output[0]))
                                output_min = float(min(output[0]))
                                #print(output_max)
                                #print(output_min)
                                output_delta = float(output_max - output_min)

                                tmp_series = pd.Series([i,input_ori[0],input_ori[1],input_ori[2],input_ori[3],input_ori[4],input_ori[5],output[0][0],output[0][1],output[0][2],output[0][3],output_max,output_min,output_delta])

                                if output_delta < output_delta_temp * 1.05:
                                    output_delta_temp = min(output_delta,output_delta_temp)
                                    
                                    predict_df = predict_df.append(tmp_series,ignore_index = True)
                                
                                i +=1
                                #if i > 100 :
                                #    break

                            end_time = time.time()
                            total_time = end_time - start_time
                            print('loop time is ', total_time)
                            '''

                            predict_df_s = iter_deeplearning_df.sort_values('Tdelta')

                            predict_df_s.to_csv('C:\deeplearning/model/predict/'
                                                 + savename
                                                 + '_predict.csv')
            
        else:
            layer_size=62
            bn_where=1
            keep_prob=0.2
            for ac_last in [1]:
                model = get_model(num_layers, layer_size, bn_where, ac_last, keep_prob)

                model.fit(input_train, output_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(input_test, output_test),
                          callbacks=[es_cb])

                score_test = model.evaluate(input_test, output_test, verbose=0)
                score_train = model.evaluate(input_train, output_train, verbose=0)
                test_predict = model.predict(input_test, batch_size=32, verbose=1)
                train_predict = model.predict(input_train, batch_size=32, verbose=1)

                df_test_pre = pd.DataFrame(test_predict)
                df_train_pre = pd.DataFrame(train_predict)

                df_test_param = pd.DataFrame(output_test)
                df_train_param = pd.DataFrame(output_train)

                df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                savename = ""
                savename = savename  + '_score_train-' + str("%.3f" % round(math.log10(score_train[0]), 3))
                savename = savename  + '_score_test-' + str("%.3f" % round(math.log10(score_test[0]), 3))
                savename = savename  + '_numlayer-' + str(num_layers)
                savename = savename  + '_layersize-' + str(layer_size)
                savename = savename  + '_bn- ' + str(bn_where)
                savename = savename  + '_ac-' + str(ac_last)
                savename = savename  + '_k_p-' + str(keep_prob)
                savename = savename  + '_patience-' + str(patience_)


                df_dens_test.to_csv('C:\deeplearning/model/csv/'
                                       + savename
                                       + '_test.csv')

                df_dens_train.to_csv('C:\deeplearning/model/csv/'
                                       + savename
                                       + '_train.csv')


                model.save('C:\Deeplearning/model/8ken/model_8ken'
                                       + savename
                                       + '.h5')
                # plot_model(model, to_file='C:\Deeplearning/model.png')
                #plot_model(model, to_file='model.png')


                print('Test loss:', score_test[0])
                print('Test accuracy:', score_test[1])

                #print('predict', test_predict)

                print('C:\Deeplearning/model/8ken/model_8ken'
                               + savename
                               + '.h5')

                model.summary()
