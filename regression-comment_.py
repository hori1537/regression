

# programmed by YUKI Horie, Glass Research Center 'yuki.horie@'
# do not use JAPANESE!

from __future__ import print_function

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import sklearn.ensemble
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import _tree
from sklearn.externals.six import StringIO
from sklearn import linear_model
from sklearn import svm
#ver0.20
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


import numpy as np
import pandas as pd
import math
import time
import random
import os
import sys
import itertools
import h5py
import copy


import matplotlib.pyplot as plt 
import seaborn as sns

#import dtreeviz.trees
#import dtreeviz.shadow
#import dtreeviz

import pydotplus

import xgboost as xgb
#from xgboost import plot_tree # sklearn has also plot_tree, so do not import plot_Tree


# chkprint 
from inspect import currentframe
def chkprint(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

# get variable name
def get_variablename(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    return '_' + ', '.join(names.get(id(arg),'???') + '_' + repr(arg) for arg in args)


# fix the np.random.seed, it can get the same results every time to run this program
np.random.seed(1)


# check the desktop path and move to the desktop path
desktop_path =  os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop"
os.chdir(desktop_path)

# Graphviz path
#http://spacegomi.hatenablog.com/entry/2018/01/26/170721
sys.path.append('C:\\Users\\1310202\\AppData\\Local\\Continuum\\anaconda3\\envs\\py36\\Library\\bin\\graphviz')

graphviz_path = 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe'
graphviz_path = 'C:\\Users\\1310202\\Desktop\\software\\graphviz-2.38\\release\\bin\\dot.exe'

# name of theme name
theme_name = 'random'

# outputs of dataset
address_ = 'C:/Users/1310202/Desktop/20180921/horie/data_science/for8ken/'
address_ = 'C:/Users/1310202/Desktop/20180921/horie/data_science/'
#address_ = 'E:/8ken/'

# Date from COMSOL calculated by FUJII TAKESHI and SAITO MASANORI Glass Research Center
CSV_NAME = "inputcsv_20181011-.csv"
CSV_NAME = "rand.csv"

# csv information
info_num    = 1 # information columns in csv file
input_num   = 1 # input data  columns in csv file
output_num  = 1 # output data columns in csv file

info_col    = info_num 
input_col   = info_num + input_num 
output_col  = info_num + input_num + output_num

# evaluate of all candidate or not
is_gridsearch = True

# perform deeplearning or not
is_dl = False


if os.path.exists(address_) == False:
    print('no exist ', address_)

# make the save folder
print('makedir ', 'ML')

#save_address = 'ML\\' + theme_name +'\\'


# make the folder for saving the results
def chk_mkdir(paths):
    for path_name in paths:
        if os.path.exists(path_name) == False:
            os.mkdir(path_name)
    return
    

paths = [ 'ML',
          'ML\\' + theme_name, 
          'ML\\' + theme_name + '\\sklearn', 
          'ML\\' + theme_name + '\\sklearn\\tree', 
          'ML\\' + theme_name + '\\sklearn\\importance',
          'ML\\' + theme_name + '\\sklearn\\parameter',
          'ML\\' + theme_name + '\\sklearn\\predict',
          'ML\\' + theme_name + '\\sklearn\\traintest',
          'ML\\' + theme_name + '\\deeplearning',
          'ML\\' + theme_name + '\\deeplearning\\h5',
          'ML\\' + theme_name + '\\deeplearning\\traintest',
          'ML\\' + theme_name + '\\deeplearning\\predict']

chk_mkdir(paths)
os.chdir('ML')
os.chdir(theme_name)


#read the dataset

np.random.seed(1)

#raw_data_df = pd.read_csv(open(str(address_) + str(CSV_NAME) ,encoding="utf-8_sig"))
raw_data_df = pd.read_csv(open(str(address_) + str(CSV_NAME)))


info_df     = raw_data_df.iloc[:, 0         : info_col]
input_df    = raw_data_df.iloc[:, info_col  : input_col]
output_df   = raw_data_df.iloc[:, input_col : output_col]


input_des   = input_df.describe() 
output_des  = output_df.describe() 

input_max   = input_des.loc['max']
input_min   = input_des.loc['min']
output_max  = output_des.loc['max']
output_min  = output_des.loc['min']


info_feature_names  = info_df.columns
input_feature_names = input_df.columns
output_feature_names= output_df.columns
predict_output_feature_names = list(map(lambda x:x + '-predict' , output_feature_names))


chkprint(info_feature_names)
chkprint(input_feature_names)
chkprint(output_feature_names)
chkprint(predict_output_feature_names)

from sklearn.preprocessing import StandardScaler
all_sc_model    = StandardScaler()
input_sc_model  = StandardScaler()
output_sc_model = StandardScaler()

raw_data_std_df = all_sc_model.fit_transform(raw_data_df)
input_std_df    = input_sc_model.fit_transform(input_df)
output_std_df   = output_sc_model.fit_transform(output_df)


# split train data and test data from the raw_data_std_df
train_std_df, test_std_df = train_test_split(raw_data_std_df, test_size=0.1)
train_df, test_df = train_test_split(raw_data_df, test_size=0.1)

# transform from pandas dataframe to numpy array
train_np = np.array(train_df)
test_np  = np.array(test_df)
train_std_np = np.array(train_std_df)
test_std_np  = np.array(test_std_df)



# split columns to info, input, output
[train_info, train_input, train_output] = np.hsplit(train_np, [info_col, input_col])
[test_info,  test_input,  test_output]  = np.hsplit(test_np,  [info_col, input_col])

[train_info, train_input_std, train_output_std] = np.hsplit(train_std_np, [info_col, input_col])
[test_info,  test_input_std,  test_output_std]  = np.hsplit(test_std_np,  [info_col, input_col])

train_input_df  = pd.DataFrame(train_input, columns = input_feature_names)
test_input_df   = pd.DataFrame(test_input,  columns = input_feature_names)
train_output_df = pd.DataFrame(train_output,columns = output_feature_names)
test_output_df  = pd.DataFrame(test_output, columns = output_feature_names)

train_input_std_df  = pd.DataFrame(train_input_std, columns = input_feature_names)
test_input_std_df   = pd.DataFrame(test_input_std,  columns = input_feature_names)
train_output_std_df = pd.DataFrame(train_output_std,columns = output_feature_names)
test_output_std_df  = pd.DataFrame(test_output_std, columns = output_feature_names)



# detect the outliner by OneClassSVM      not use now
#from sklearn.svm import OneClassSVM
#ocsvm  = OneClassSVM(nu=0.01)





# select the feature value by the random forest regressor
max_depth = 7
model       = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
model_name  = ''
model_name  += 'RandomForestRegressor_'
model_name  += 'max_depth_'+str(max_depth)

model.fit(train_input, train_output)
importances = np.array(model.feature_importances_)
importances_sort = importances.argsort()[::-1]
split_base  = np.array([15,13,9,4,4,3,3,3]) # max:758160
split_base  = np.array([10,7,3,3,3,3,3,3])  # max:51030

# set the split num from importances rank of random forest regressor
split_num   = np.full(len(importances_sort),1)
chkprint(split_num)
chkprint(split_base)
for i in range(len(importances)):
    rank_ = importances_sort[i]
    chkprint(rank_)
    split_num[rank_] = split_base[i]

chkprint(split_num)


def combination(max, min, split_num):
    candidate = []
    for i in range(input_num):
        candidate.append(np.linspace(start = max[i], stop = min[i], num = split_num[i]))

    candidate = np.array(candidate)        
    return candidate


if is_gridsearch == True:

    ### manual setting of the split num of grid search candidate
    
    #split_num = np.full((input_num), 2)
    #split_num = np.array([1, 1, 1, 2, 3, 4])
    all_gridsearch_number = split_num.prod()
    chkprint(all_gridsearch_number)

    if all_gridsearch_number>10**7:
        print('so much candidate')
        print('Do you really want to predict ', all_gridsearch_number, ' candidate? y/n')
        answer = input()
        if answer in ['yes','ye','y']:
            print('ok lets start')
        else:
            print('exit the program')
            sys.exit()

    if all_gridsearch_number>10**10:
        print('but, too much candidate')
        print('exit the program')
        sys.exit()


    candidate = combination(input_max, input_min, split_num)

    # refer from https://teratail.com/questions/152110
    # unpack   *candidate
    gridsearch_input        = list(itertools.product(*candidate))
    gridsearch_input_std    = input_sc_model.transform(gridsearch_input)

    gridsearch_input_df     = pd.DataFrame(gridsearch_input, columns = input_feature_names)
    gridsearch_input_std_df = pd.DataFrame(gridsearch_input_std, columns = input_feature_names)
    
    #all_gridsearch_number = sum(len(i) for i in gridsearch_input)
    #chkprint(all_gridsearch_number)


    ############### iterate the all candidate #######################
    #gridsearch_input = list(itertools.product(xpot_candidate, ypot_candidate, tend_candidate, tside_candidate, tmesh_candidate, hter_candidate))
    #gridsearch_input = np.reshape(gridsearch_input,[candidate_number, input_col])
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

#########   predict of all candidate by the scikitlearn model ###############

def gridsearch_predict(model_std, model_name):        
    start_time = time.time()

    gridsearch_predict_std_df   = pd.DataFrame(model_std.predict(gridsearch_input_std), columns = predict_output_feature_names)
    gridsearch_predict_df       = pd.DataFrame(output_sc_model.inverse_transform(gridsearch_predict_std_df), columns = predict_output_feature_names)

    end_time = time.time()
    total_time = end_time - start_time
    print(model_name, ' ' , total_time)

    gridsearch_predict_df['Tmax-']      = gridsearch_predict_df.max(axis = 1)
    gridsearch_predict_df['Tdelta-']    = gridsearch_predict_df.min(axis = 1)
    gridsearch_predict_df['Tmin-']      = gridsearch_predict_df['Tmax-'] - gridsearch_predict_df['Tdelta-']
    
    gridsearch_predict_df = pd.concat([gridsearch_input_df, gridsearch_predict_df], axis = 1)
    gridsearch_predict_df.to_csv('sklearn/predict/' + str(model_name) + '_predict.csv')

    return
        
#########   search by hyperopt ###############
#import hyperopt


#########   regression by the scikitlearn model ###############
columns_results = ['model_name', 'train_model_mse', 'train_model_rmse', 'test_model_mse', 'test_model_rmse', 'train_model_score', 'test_model_score']
allmodel_results_df = pd.DataFrame(columns = columns_results)
print(allmodel_results_df)
    
def regression(model, model_name, train_input_std, test_input_std):

    start_time = time.time()
    
    model_raw = model
    model_std = copy.deepcopy(model)
    
    model_raw.fit(train_input, train_output)
    model_std.fit(train_input_std, train_output_std)

    save_regression(model_raw, model_std, model_name, train_input_std, test_input_std)

    return


def save_regression(model_raw, model_std, model_name, train_input_std, test_input_std):
    global allmodel_results_df

    train_output_predict_std    = model_std.predict(train_input_std)
    test_output_predict_std     = model_std.predict(test_input_std)
    train_output_predict        = output_sc_model.inverse_transform(train_output_predict_std)
    test_output_predict         = output_sc_model.inverse_transform(test_output_predict_std)

    if hasattr(model_std, 'score') == True:
        train_model_score   = model_std.score(train_input_std , train_output_std)
        train_model_score   = np.round(train_model_score,3)
        test_model_score    = model_std.score(test_input_std,   test_output_std)
        test_model_score    = np.round(test_model_score,3)
    if hasattr(model_std, 'evaluate') == True:
        train_model_score   = model_std.evaluate(train_input_std , train_output_std)
        train_model_score   = np.round(train_model_score,3)
        test_model_score    = model_std.evaluate(test_input_std,   test_output_std)
        test_model_score    = np.round(test_model_score,3)

    train_output_predict_df = pd.DataFrame(train_output_predict, columns = predict_output_feature_names)
    train_result_df         = pd.concat([train_input_df, train_output_predict_df, train_output_df], axis=1)

    test_output_predict_df  = pd.DataFrame(test_output_predict, columns = predict_output_feature_names)
    test_result_df          = pd.concat([train_input_df, test_output_predict_df, train_output_df], axis=1)
    
    train_model_mse     = sklearn.metrics.mean_squared_error(train_output_std, train_output_predict_std)
    train_model_rmse    = np.sqrt(train_model_mse)
    test_model_mse      = sklearn.metrics.mean_squared_error(test_output_std, test_output_predict_std)
    test_model_rmse     = np.sqrt(test_model_mse)
    

    results_df = pd.DataFrame([model_name, train_model_mse, train_model_rmse, test_model_mse, test_model_rmse, train_model_score, test_model_score]).T
    results_df.columns = columns_results
    allmodel_results_df = pd.concat([allmodel_results_df, results_df])

    
    model_name += get_variablename(train_model_score,test_model_score)
    
    chkprint(model_name)
    
    train_result_df.to_csv('sklearn/traintest/' + str(model_name) + '_train.csv')
    test_result_df.to_csv('sklearn/traintest/' + str(model_name) + '_test.csv')
    
    if hasattr(model_std, 'get_params') == True:
        model_params          = model_std.get_params()
        params_df = pd.DataFrame([model_std.get_params])
    
    if hasattr(model_std, 'intercept_') == True &  hasattr(model_std, 'coef_') == True:
        model_intercept_df    = pd.DataFrame(model_std.intercept_)
        model_coef_df         = pd.DataFrame(model_std.coef_)
        model_parameter_df    = pd.concat([model_intercept_df, model_coef_df])
        model_parameter_df.to_csv('sklearn/parameter/' + str(model_name) + '_parameter.csv')
        
    if (hasattr(model, 'get_booster')):
            # HACK https://github.com/dmlc/xgboost/issues/1238
            print(model.get_booster().feature_names)
            sys.exit()
    
    if hasattr(model_raw, 'tree_') == True:
        try:
            #tree_to_code(model_raw, [input_feature_names])
            
            dot_data = StringIO()
            sklearn.tree.export_graphviz(model_raw, out_file=dot_data, feature_names=input_feature_names)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            
            # refer from https://qiita.com/wm5775/items/1062cc1e96726b153e28
            # the Graphviz2.38 dot.exe
            graph.progs = {'dot':graphviz_path}
            

            graph.write_pdf('sklearn/tree/' + 'decisiontree' + str(max_depth) + '.pdf')
        except:
            print('Error occured about tree to code. Check the graphviz_path')
            pass
            


        #viz = dtreeviz.trees.dtreeviz(model, train_input, train_output, target_name = output_feature_names, feature_names = input_feature_names, X = test_input)
        try:
            viz = dtreeviz.trees.dtreeviz(model,
                                      train_input,
                                      train_output, 
                                      target_name = output_feature_names, 
                                      feature_names = input_feature_names)

            viz.save("boston.svg")
            viz.view()
            sys.exit()
        except:
            pass
            
    if hasattr(model_raw, 'feature_importances_') == True:

        importances = pd.Series(model_raw.feature_importances_)
        importances = np.array(importances)
        print(importances)
        #importances = importances.sort_values()
        
        label       = input_feature_names
        plt.bar(label, importances)
        
        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)
        plt.rcParams["font.size"] = 12

        plt.title("importance in the tree " + str(theme_name))
        #plt.show()
        plt.savefig('sklearn/importance/' + str(model_name)  + '.png', dpi = 240)

    # call gridsearch_predict 
    if is_gridsearch == True:
        gridsearch_predict(model_std, model_name)
    
    return


##################### Linear Regression #####################

model = linear_model.LinearRegression()

model_name = 'linear_regression_'

regression(model, model_name, train_input_std, test_input_std)

##################### Regression of Stochastic Gradient Descent ##################### 
max_iter = 1000

model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter = max_iter))
model_name = 'Stochastic Gradient Descent_'
model_name += 'max_iter_'+str(max_iter)


regression(model, model_name, train_input_std, test_input_std)


##################### Regression of SVR #####################
kernel_ = 'rbf'
C_= 1

model = MultiOutputRegressor(svm.SVR(kernel = kernel_, C = C_))
model_name = 'SupportVectorRegressor_'
model_name += 'kernel_'+str(kernel_)
model_name += 'C_'+str(C_)

regression(model, model_name, train_input_std, test_input_std)

##################### Regression of Ridge #####################
alpha_ = 1.0
model = linear_model.Ridge(alpha = alpha_)
model_name = 'Ridge_'
model_name += 'alpha_'+str(alpha_)

regression(model, model_name, train_input_std, test_input_std)


##################### Regression of Lasso #####################
alpha_ = 1.0
model = linear_model.Lasso(alpha = alpha_)
model_name = 'Lasso_'
model_name += 'alpha_'+str(alpha_)

regression(model, model_name, train_input_std, test_input_std)


##################### Regression of Elastic Net #####################
alpha_ =1.0
l1_ratio_ = 0.5
model = linear_model.ElasticNet(alpha=alpha_, l1_ratio = l1_ratio_)
model_name = 'ElasticNet_'
model_name += 'alpha_'+str(alpha_)
model_name += 'l1_ratio_'+str(l1_ratio_)

regression(model, model_name, train_input_std, test_input_std)


##################### Regression of MultiTaskLassoCV #####################
max_iter_ = 1000 
model = linear_model.MultiTaskLassoCV()
model_name = 'MTLasso_'
model_name += 'max_iter_'+str(max_iter)

regression(model, model_name, train_input_std, test_input_std)

##################### Regression of Multi Task Elastic Net CV #####################
model = linear_model.MultiTaskElasticNetCV()

model_name = 'MTElasticNet_'


##################### Regression of OrthogonalMatchingPursuit #####################
model = linear_model.OrthogonalMatchingPursuit()
model_name = 'OrthogonalMatchingPursuit_'

regression(model, model_name, train_input_std, test_input_std)

##################### Regression of BayesianRidge #####################
model = MultiOutputRegressor(linear_model.BayesianRidge())
model_name = 'BayesianRidge_'

regression(model, model_name, train_input_std, test_input_std)

##################### Regression of PassiveAggressiveRegressor #####################
#model = MultiOutputRegressor(linear_model.PassiveAggressiveRegressor())
#model_name = 'PassiveAggressiveRegressor_'

#regression(model, model_name, train_input,train_output)

##################### Regression of PolynomialFeatures #####################
'''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
# http://techtipshoge.blogspot.com/2015/06/scikit-learn.html
# http://enakai00.hatenablog.com/entry/2017/10/13/145337

for degree in [2]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree = 2),
        'linear', MultiOutputRegressor(linear_model.LinearRegression()))
        ])
    
    model_name = 'PolynomialFeatures_'
    model_name += 'degree_' + str(degree)
    
    regression(model, model_name, train_input,train_output)
'''

##################### Regression ofGaussianProcessRegressor #####################
'''
from sklearn.gaussian_process import GaussianProcessRegressor

model = MultiOutputRegressor(GaussianProcessRegressor())
model_name = 'GaussianProcessRegressor_'
    
regression(model, model_name, train_input,train_output)
'''

##################### Regression of GaussianNB #####################

'''
from sklearn.naive_bayes import GaussianNB

model = MultiOutputRegressor(GaussianNB())
model_name = 'GaussianNB_'

regression(model, model_name, train_input,train_output)
'''
##################### Regression of GaussianNB #####################
'''
from sklearn.naive_bayes import  ComplementNB

model = ComplementNB()
model_name = 'ComplementNB_'
    
regression(model, model_name, train_input,train_output)
'''
##################### Regression of MultinomialNB #####################
'''
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model_name = 'MultinomialNB_'
    
regression(model, model_name, train_input,train_output)
'''

##################### Regression of DecisionTreeRegressor #####################
for max_depth in [4,5,6,7,8,9]:
    model = sklearn.tree.DecisionTreeRegressor(max_depth = max_depth)
    model_name = 'DecisionTreeRegressor_'    
    model_name += 'max_depth_'+str(max_depth)

    regression(model, model_name, train_input_std, test_input_std)

#################### Regression of RandomForestRegressor #####################
for max_depth in [4,5,6,7,8,9]:
    model = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
    model_name = ''
    model_name += 'RandomForestRegressor_'
    #model_name += get_variablename(max_depth)
    model_name += 'max_depth_'+str(max_depth)
    
    regression(model, model_name, train_input_std, test_input_std)

##################### Regression of XGBoost #####################
# refer from https://github.com/FelixNeutatz/ED2/blob/23170b05c7c800e2d2e2cf80d62703ee540d2bcb/src/model/ml/CellPredict.py

estimator__min_child_weight_ = [1,5] #1,3 
estimator__subsample_        = [0.9] #0.7, 0.8, 
estimator__learning_rate_    = [0.1,0.01] #0.1
estimator__max_depth_        = [3, 7]
estimator__n_estimators_      = [100,1000]

for estimator__min_child_weight, estimator__subsample, estimator__learning_rate, estimator__max_depth, estimator__n_estimators \
     in itertools.product(estimator__min_child_weight_, estimator__subsample_, estimator__learning_rate_, estimator__max_depth_,estimator__n_estimators_ ):

    xgb_params = {'estimator__min_child_weight': estimator__min_child_weight,
                  'estimator__subsample': estimator__subsample,
                  'estimator__learning_rate': estimator__learning_rate,
                  'estimator__max_depth': estimator__max_depth,
                  'estimator__n_estimators': estimator__n_estimators,
                  'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'reg:linear'}

    model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
    
    model_name = 'XGBoost'
    model_name += 'min_child_weight_'+str(estimator__min_child_weight)
    model_name += 'subsample_'+str(estimator__subsample)
    model_name += 'learning_rate_'+str(estimator__learning_rate)
    model_name += 'max_depth_'+str(estimator__max_depth)
    model_name += 'n_estimators_'+str(estimator__n_estimators)

    regression(model, model_name, train_input_std, test_input_std)



################# to csv ##############################
allmodel_results_df.to_csv('comparison of methods.csv')

#######################################################




'''
################# importances feature by XGBOOST ######
import matplotlib.pyplot as plt

importances = pd.Series(reg1_multigbtree.feature_importances_)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()
#######################################################
'''




'''
##################### LIME Explainer #####################
import lime
import lime.lime_tabular

#explainer1 = lime.lime_tabular.LimeTabularExplainer(train_output, feature_names=input_feature_names, kernel_width=3)
0
explainer1 = lime.lime_tabular.LimeTabularExplainer(train_input, feature_names= input_feature_names, class_names=output_feature_names, verbose=True, mode='regression')

np.np.random.seed(1)
i = 3
#exp = explainer.explain_instance(test[2], predict_fn, num_features=10)
exp = explainer.explain_instance(test[i], reg1_SVR.predict, num_features=5)

sys.exit()
# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path= str(address_) + 'numeric_category_feat_01', show_table=True, show_all=True)

i = 3
exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
# exp.show_in_notebook(show_all=False)
exp.save_to_file(file_path=str(address_) + 'numeric_category_feat_02', show_table=True, show_all=True)
##########################################################
'''




'''
# import pickle
# pickle.dump(reg, open("model.pkl", "wb"))
# reg = pickle.load(open("model.pkl", "rb"))

pred1_train = reg1_gbtree.predict(train_input)
pred1_test = reg1_gbtree.predict(test_input)
print(mean_squared_error(train_output, pred1_train))
print(mean_squared_error(test_output, pred1_test))

import matplotlib.pyplot as plt

importances = pd.Series(reg1_gbtree.feature_importances_)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()
'''



##################### Deep Learning #####################
if is_dl == False:
    sys.exit()


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
from keras.wrappers.scikit_learn import KerasRegressor

from keras.models import load_model
from keras.utils import plot_model



def get_model(num_layers, layer_size,bn_where,ac_last,keep_prob, patience):
    
    model =Sequential()
    model.add(InputLayer(input_shape=(input_num,)))
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

    model.add(Dense(output_num))


    model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    return model

def get_model2(num_layers, layer_size,bn_where,ac_last,keep_prob, patience):
    model = Sequential()
    model.add(Dense(input_num, input_dim = input_num, activation = 'relu'))

    for i in range(num_layers):
        model.add(Dense(layer_size, activation = 'relu'))
    
    model.add(BatchNormalization(mode=0))
    model.add(Dense(output_num))

    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    return model
# refer from https://github.com/completelyAbsorbed/ML/blob/0ca17d25bae327fe9be8e3639426dc86f3555a5a/Practice/housing/housing_regression_NN.py


num_layers  = [4,5]
layer_size  = [64, 32, 16]
bn_where    = [3, 0]
ac_last     = [0, 1]
keep_prob   = [0]
patience    = [3000]

'''
for dp_params in itertools.product(num_layers, layer_size, bn_where, ac_last, keep_prob, patience):
    num_layers, layer_size, bn_where, ac_last, keep_prob, patience = dp_params
    
    batch_sixe  = 30
    nb_epochs   = 10000
    cb = keras.callbacks.EarlyStopping(monitor = 'loss'   , min_delta = 0,
                                 patience = patience, mode = 'auto')
                                 
    model = KerasRegressor(build_fn = get_model(*dp_params), nb_epoch=5000, batch_size=5, verbose=0, callbacks=[cb])
    
    model_name =   'deeplearning'
    model_name +=  '_numlayer-'       + str(num_layers)
    model_name +=  '_layersize-'      + str(layer_size)
    model_name +=  '_bn- '            + str(bn_where)
    model_name +=  '_ac-'             + str(ac_last)
    model_name +=  '_k_p-'            + str(keep_prob)
    model_name +=  '_patience-'       + str(patience)
    
    regression(model, model_name, train_input, train_output)
'''


allmodel_results_df.to_csv('comparison of methods.csv')

epochs = 100000
batch_size = 32

for patience_ in [100,3000]:

    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_, verbose=0, mode='auto')

    for num_layers in [4,3,2]:
        if num_layers !=1:
            for layer_size in[1024,512,256,128,64,32,16]:
                for bn_where in [3,0,1,2]:
                    for ac_last in [0,1]:
                        for keep_prob in [0,0.1,0.2]:

                            model =get_model(num_layers,layer_size,bn_where,ac_last,keep_prob, patience)
                            #model = KerasRegressor(build_fn = model, epochs=5000, batch_size=5, verbose=0, callbacks=[es_cb])

                            if layer_size >= 1024:
                                batch_size = 30
                            elif num_layers >= 4:
                                batch_size = 30
                            elif bn_where ==3:
                                batch_size=30

                            else:
                                batch_size = 30

                            model_name = "deeplearning"
                            model_name +=  '_numlayer-'       + str(num_layers)
                            model_name +=  '_layersize-'      + str(layer_size)
                            model_name +=  '_bn- '            + str(bn_where)
                            model_name +=  '_ac-'             + str(ac_last)
                            model_name +=  '_k_p-'            + str(keep_prob)
                            model_name +=  '_pat-'       + str(patience_)

                            regression(model, model_name, train_input_std, test_input_std)



                            
                            model.fit(train_input, train_output,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=(test_input, test_output),
                                      callbacks=[es_cb])
                            
                            save_regression(model, model_name, train_input_std, test_input_std)
                            
                            '''
                            score_test = model.evaluate(test_input, test_output, verbose=1)
                            score_train = model.evaluate(train_input, train_output, verbose=1)
                            test_predict = model.predict(test_input, batch_size=32, verbose=1)
                            train_predict = model.predict(train_input, batch_size=32, verbose=1)

                            df_test_pre = pd.DataFrame(test_predict)
                            df_train_pre = pd.DataFrame(train_predict)

                            df_test_param = pd.DataFrame(test_output)
                            df_train_param = pd.DataFrame(train_output)

                            df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                            df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                            savename = ""
                            savename +=  '_score_train-'    + str("%.3f" % round(score_train[0],3))
                            savename +=  '_score_test-'     + str("%.3f" % round((score_test[0]),3))
                            savename +=  '_numlayer-'       + str(num_layers)
                            savename +=  '_layersize-'      + str(layer_size)
                            savename +=  '_bn- '            + str(bn_where)
                            savename +=  '_ac-'             + str(ac_last)
                            savename +=  '_k_p-'            + str(keep_prob)
                            savename +=  '_patience-'       + str(patience_)

                            df_dens_test.to_csv('deeplearning/traintest/' + savename + '_test.csv')
                            df_dens_train.to_csv('deeplearning/traintest/' + savename + '_train.csv')
                            '''

                            model.save('deeplearning/h5/' + model_name + '.h5')

                            
                            model.summary()

                            
                            '''
                            ### evaluation of deeplearning ###
                            def eval_bydeeplearning(input):
                                output_predict = model.predict(input, batch_size = 1, verbose= 1)
                                output_predict = np.array(output_predict)

                                return output_predict

                            if is_gridsearch == True:

                                gridsearch_output = eval_bydeeplearning(gridsearch_input)
                                
                
                                print('start the evaluation by deeplearning')
                                print('candidate is ', candidate_number)
                                
                                start_time = time.time()
                                                           
                                iter_deeplearning_predict_df = pd.DataFrame(gridsearch_output, columns = predict_output_feature_names)                            
                                iter_deeplearning_predict_df['Tmax'] = iter_deeplearning_predict_df.max(axis=1)
                                iter_deeplearning_predict_df['Tmin'] = iter_deeplearning_predict_df.min(axis=1)
                                iter_deeplearning_predict_df['Tdelta'] = iter_deeplearning_predict_df['Tmax'] -  iter_deeplearning_predict_df['Tmin']

                                iter_deeplearning_df = pd.concat([gridsearch_input_std_df, iter_deeplearning_predict_df], axis=1)

                                end_time = time.time()

                                total_time = end_time - start_time
                                print('total_time 1', total_time)

                                predict_df_s = iter_deeplearning_df.sort_values('Tdelta')

                                predict_df_s.to_csv('deeplearning/predict/'
                                                    + savename
                                                    + '_predict.csv')

                                # evaluate by the for - loop     Not use now
                                
                                
                                i=0                            
                                predict_df = pd.DataFrame()
                                output_delta_temp = 100000
                            
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

            
        else:
            layer_size=62
            bn_where=1
            keep_prob=0.2
            for ac_last in [1]:
                model = get_model(num_layers, layer_size, bn_where, ac_last, keep_prob)

                model.fit(train_input, train_output,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(test_input, test_output),
                          callbacks=[es_cb])

                score_test = model.evaluate(test_input, test_output, verbose=0)
                score_train = model.evaluate(train_input, train_output, verbose=0)
                test_predict = model.predict(test_input, batch_size=32, verbose=1)
                train_predict = model.predict(train_input, batch_size=32, verbose=1)

                df_test_pre = pd.DataFrame(test_predict)
                df_train_pre = pd.DataFrame(train_predict)

                df_test_param = pd.DataFrame(test_output)
                df_train_param = pd.DataFrame(train_output)

                df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                savename = ""
                savename +=  '_score_train-' + str("%.3f" % round(math.log10(score_train[0]), 3))
                savename +=  '_score_test-' + str("%.3f" % round(math.log10(score_test[0]), 3))
                savename +=  '_numlayer-' + str(num_layers)
                savename +=  '_layersize-' + str(layer_size)
                savename +=  '_bn- ' + str(bn_where)
                savename +=  '_ac-' + str(ac_last)
                savename +=  '_k_p-' + str(keep_prob)
                savename +=  '_patience-' + str(patience_)


                df_dens_test.to_csv('deeplearning/traintest/' + savename + '_test.csv')

                df_dens_train.to_csv('deeplearning/traintest/' + savename + '_train.csv')


                model.save('deeplearning/h5/'
                                       + savename
                                       + '.h5')
                # plot_model(model, to_file='C:\Deeplearning/model.png')
                #plot_model(model, to_file='model.png')

                print('Test loss:', score_test[0])
                print('Test accuracy:', score_test[1])

                #print('predict', test_predict)

                model.summary()                

