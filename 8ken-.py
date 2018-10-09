

from __future__ import print_function

# programmed by Horie, Glass Research Center 'yuki.horie@'

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
import math
import time
import pandas as pd
import pandas 
import random
import os
import sys
import itertools
import h5py

import pydotplus

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

# chkprint 
from inspect import currentframe
def chkprint(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

# save 
def get_variablename(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    return '_' + ', '.join(names.get(id(arg),'???') + '_' + repr(arg) for arg in args)

random.seed(1)

desktop_path =  os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop"
chkprint(desktop_path)
os.chdir(desktop_path)


# name of themkene
theme_name = '8ken'

# outputs of dataset
address_ = 'C:/Users/1310202/Desktop/20180921/horie/data_science/for8ken/'
save_address = desktop_path + 'ML'


# Date from COMSOL calculated by FUJII TAKESHI Glass Research Center
CSV_NAME="inputcsv_simulation.csv"
DATA_SIZE = 144

# make the dir

if os.path.exists(address_) == False:
    print('no exist ', address_)

if os.path.exists('ML') == False:
    print('makedir ', 'ML')
    
    try:
        os.mkdir('ML')
        os.chdir('ML')
        if os.path.exists(theme_name) == False:
            os.mkdir(theme_name)
            os.chdir(theme_name)
        if os.path.exists('sklearn') == False:
            os.mkdir('sklearn')
        if os.path.exists('deeplearning') == False:
            os.mkdir('deeplearning')
            os.mkdir('deeplearning\\h5')
            os.mkdir('deeplearning\\csv')
    except:
        print("Failed to mkdir")
else:
    os.chdir('ML')
    os.chdir(theme_name)
    
    
info_num    = 0
input_num   = 6 
output_num  = 4

info_col    = info_num 
input_col   = info_num + input_num 
output_col  = info_num + input_num + output_num



#read the dataset
raw_data_df = pandas.read_csv(open(str(address_) + str(CSV_NAME) ,encoding="utf-8_sig"))

info_df     = raw_data_df.iloc[:, 0         : info_col]
input_df    = raw_data_df.iloc[:, info_col  : input_col]
output_df   = raw_data_df.iloc[:, input_col : output_col]

info_feature_names  = info_df.columns
input_feature_names = input_df.columns
output_feature_names= output_df.columns

chkprint(info_feature_names)
chkprint(input_feature_names)
chkprint(output_feature_names)


train_df, test_df = train_test_split(raw_data_df, test_size=0.1)

train_np = np.array(train_df)
test_np  = np.array(test_df)
 
#dataset
[info_train, input_train, output_train, Tmax_train, Tmin_train, Tdelta_train] = np.hsplit(train_np, [info_col, input_col, output_col, 11, 12])
[info_test,  input_test,  output_test,  Tmax_test,  Tmin_test,  Tdelta_test ] = np.hsplit(test_np,  [info_col, input_col, output_col, 11, 12])


chkprint(info_test)
chkprint(input_test)
chkprint(output_test)


train_input_df  = pd.DataFrame(input_train, columns     = input_feature_names)
test_input_df   = pd.DataFrame(input_test, columns      = input_feature_names)

train_output_df = pd.DataFrame(output_train, columns    = output_feature_names)
test_output_df  = pd.DataFrame(output_test, columns     = output_feature_names)


### the range of each input
xpot_coef = 100
xpot_min = 80 / xpot_coef
xpot_max = 100 / xpot_coef
xpot_step = 2 / xpot_coef
xpot_candidate = np.arange(start = xpot_min , stop = xpot_max+ xpot_step, step = xpot_step)

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

############### iterate the all candidate #######################
input_iter = list(itertools.product(xpot_candidate, ypot_candidate, tend_candidate, tside_candidate, tmesh_candidate, hter_candidate))
input_iter = np.reshape(input_iter,[candidate_number, input_col])
iter_input_df = pd.DataFrame(input_iter, columns = input_feature_names)
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


#########   regression by the scikitlearn model ###############
columns_results = ['model_name', 'train_model_mse', 'train_model_rmse', 'test_model_mse', 'test_model_rmse', 'train_model_score', 'test_model_score']
allmodel_results_df = pd.DataFrame(columns = columns_results)
print(allmodel_results_df)
    
def regression(model, model_name, input_train, output_train):
    global allmodel_results_df
    model.fit(input_train, output_train)
    
    #print(model.get_params)
    

    train_model_mse = sklearn.metrics.mean_squared_error(output_train, model.predict(input_train))
    train_model_rmse = np.sqrt(train_model_mse)
    test_model_mse = sklearn.metrics.mean_squared_error(output_test, model.predict(input_test))
    test_model_rmse = np.sqrt(test_model_mse)
    
    chkprint(model_name, train_model_mse)
    chkprint(model_name, train_model_rmse)
    chkprint(model_name, test_model_mse)
    chkprint(model_name, test_model_rmse)
    
    train_model_score = model.score(input_train , output_train)
    train_model_score = np.round(train_model_score,3)
    test_model_score  = model.score(input_test,   output_test)
    test_model_score  = np.round(test_model_score,3)
    
    
    results_df = pd.DataFrame([model_name, train_model_mse, train_model_rmse, test_model_mse, test_model_rmse, train_model_score, test_model_score]).T
    results_df.columns = columns_results
    allmodel_results_df = pd.concat([allmodel_results_df, results_df])
    params_df = pd.DataFrame([model.get_params])
    
    #if iter == True:
    #    pass
    
    train_model_predict_df = pd.DataFrame(model.predict(input_train), columns = output_feature_names)
    train_model_df = pd.concat([train_input_df, train_model_predict_df, train_output_df], axis=1)

    test_model_predict_df = pd.DataFrame(model.predict(input_test), columns = output_feature_names)
    test_model_df = pd.concat([test_input_df, test_model_predict_df, test_output_df], axis=1)

    model_name += get_variablename(train_model_score,test_model_score)
    
    chkprint(model_name, train_model_score)
    chkprint(model_name, test_model_score)
    
    train_model_df.to_csv('sklearn/' + str(model_name) + '_train.csv')
    test_model_df.to_csv('sklearn/' + str(model_name) + '_test.csv')
    
    if hasattr(model, 'get_params') == True:
        model_params          = model.get_params()
        #print(model_params)
    
    if hasattr(model, 'intercept_') == True &  hasattr(model, 'coef_') == True:
        model_intercept_df    = pd.DataFrame(model.intercept_)
        model_coef_df         = pd.DataFrame(model.coef_)
        model_parameter_df    = pd.concat([model_intercept_df, model_coef_df])
        model_parameter_df.to_csv('sklearn/' + str(model_name) + '_parameter.csv')
    

        
    if hasattr(model, 'tree_') == True:
        #tree_to_code(model, [input_feature_names])

        import pydotplus
        from sklearn.externals.six import StringIO
        dot_data = StringIO()
        sklearn.tree.export_graphviz(model, out_file=dot_data, feature_names=input_feature_names)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        #graph.progs = {'dot': u"C:\\Program Files (x86)\\Graphviz2.38\\bin\\dot.exe"}
        graph.progs = {'dot': u"C:\\Users\\1310202\\Desktop\\software\\graphviz-2.38\\release\\bin\\dot.exe"}
        # refer from https://qiita.com/wm5775/items/1062cc1e96726b153e28

        graph.write_pdf('sklearn/' + 'decisiontree' + str(max_depth) + '.pdf')

        
        
    
    return


##################### Linear Regression #####################

model = linear_model.LinearRegression()

model_name = 'linear_regression'

regression(model, model_name, input_train,output_train)

##################### Regression of Stochastic Gradient Descent ##################### 
max_iter = 1000

model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter = max_iter))
model_name = 'SGR'
model_name +=  get_variablename(max_iter)


regression(model, model_name, input_train,output_train)


##################### Regression of SVR #####################
kernel_ = 'rbf'
C_= 1

model = MultiOutputRegressor(svm.SVR(kernel = kernel_, C = C_))
model_name = 'SVR'
model_name += get_variablename(kernel_, C_)

regression(model, model_name, input_train,output_train)

##################### Regression of Ridge #####################
alpha_ = 1.0
model = linear_model.Ridge(alpha = alpha_)
model_name = 'Ridge'
model_name +=  get_variablename(alpha_)

regression(model, model_name, input_train,output_train)


##################### Regression of Lasso #####################
alpha_ = 1.0
model = linear_model.Lasso(alpha = alpha_)
model_name = 'Lasso'
model_name +=  get_variablename(alpha_)

regression(model, model_name, input_train,output_train)


##################### Regression of Elastic Net #####################
alpha_ =1.0
l1_ratio_ = 0.5
model = linear_model.ElasticNet(alpha=alpha_, l1_ratio = l1_ratio_)
model_name = 'ElasticNet'
model_name +=  get_variablename(alpha_,l1_ratio_)


regression(model, model_name, input_train,output_train)


##################### Regression of MultiTaskLassoCV #####################
max_iter_ = 1000 
model = linear_model.MultiTaskLassoCV()
model_name = 'MTLasso'
model_name +=  get_variablename(max_iter_)

regression(model, model_name, input_train,output_train)

##################### Regression of Multi Task Elastic Net CV #####################
model = linear_model.MultiTaskElasticNetCV()

model_name = 'MTElasticNet'


##################### Regression of DecisionTreeRegressor #####################
for max_depth in (1,2,3,4,5,6,7,8,9,10):
    model = sklearn.tree.DecisionTreeRegressor(max_depth = max_depth)
    model_name = 'DTR'    
    model_name += get_variablename(max_depth)

    regression(model, model_name, input_train, output_train)

##################### Regression of RandomForestRegressor #####################
for max_depth in (1,2,3,4,5):
    model = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
    model_name = ''
    model_name += 'RFR'
    model_name += get_variablename(max_depth)
    
    regression(model, model_name, input_train, output_train)

##################### Regression of XGBoost #####################
# https://github.com/FelixNeutatz/ED2/blob/23170b05c7c800e2d2e2cf80d62703ee540d2bcb/src/model/ml/CellPredict.py

estimator__min_child_weight_ = [5] #1,3 
estimator__subsample_        = [0.9] #0.7, 0.8, 
estimator__learning_rate_    = [0.01] #0.1
estimator__max_depth_        = [3, 5, 7]
estimator__n_estimators_      = [1000]

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
    
    model_name = 'XGB'
    model_name += get_variablename(estimator__min_child_weight)
    model_name += get_variablename(estimator__subsample)
    model_name += get_variablename(estimator__learning_rate)
    model_name += get_variablename(estimator__max_depth)
    model_name += get_variablename(estimator__n_estimators)
    
    regression(model, model_name, input_train, output_train)



################# to csv ##############################
allmodel_results_df.to_csv('comparison of methods.csv')
#######################################################



'''
import matplotlib.pyplot as plt

importances = pd.Series(reg1_multigbtree.feature_importances_)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()

'''






'''
##################### LIME Explainer #####################
#explainer1 = lime.lime_tabular.LimeTabularExplainer(output_train, feature_names=input_feature_names, kernel_width=3)
0
explainer1 = lime.lime_tabular.LimeTabularExplainer(input_train, feature_names= input_feature_names, class_names=output_feature_names, verbose=True, mode='regression')

np.random.seed(1)
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
                            savename +=  '_score_train-' + str("%.3f" % round(math.log10(score_train[0]), 3))
                            savename +=  '_score_test-' + str("%.3f" % round(math.log10(score_test[0]), 3))
                            savename +=  '_numlayer-' + str(num_layers)
                            savename +=  '_layersize-' + str(layer_size)
                            savename +=  '_bn- ' + str(bn_where)
                            savename +=  '_ac-' + str(ac_last)
                            savename +=  '_k_p-' + str(keep_prob)
                            savename +=  '_patience-' + str(patience_)

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
                savename +=  '_score_train-' + str("%.3f" % round(math.log10(score_train[0]), 3))
                savename +=  '_score_test-' + str("%.3f" % round(math.log10(score_test[0]), 3))
                savename +=  '_numlayer-' + str(num_layers)
                savename +=  '_layersize-' + str(layer_size)
                savename +=  '_bn- ' + str(bn_where)
                savename +=  '_ac-' + str(ac_last)
                savename +=  '_k_p-' + str(keep_prob)
                savename +=  '_patience-' + str(patience_)


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

