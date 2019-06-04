###########################################################################################
###                            predict.ALMANAC.random_guess.py                          ###
###########################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
# proj_dir = 'Z:/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
from model import data_loader
from predict_ALMANAC_DNN import dnn_model
import utility.plotting as p

###########################       function       ##############################
### mean or median guess
def random_guess(X_train, X_test, y_train, y_test, model_name, guess='mean'):
    comp_train = np.apply_along_axis(cal_index, 1, X_train)
    comp_test = np.apply_along_axis(cal_index, 1, X_test)
    # guess
    pred = dict()
    for comp in np.unique(comp_train):
        if guess == 'mean':
            pred[comp] = np.mean(y_train[comp_train == comp,:])
        elif guess == 'median':
            pred[comp] = np.median(y_train[comp_train == comp,:])
    pred_train = np.array([pred[i] for i in comp_train])
    pred_test = np.array([pred[i] for i in comp_test])
    fig_path = os.path.join(proj_dir, 'result/ALMANAC.prediction/img/{}.{}_guess.png'.format(model_name, guess))
    p.plot_correlation_train_val(y_train.reshape(-1), y_test.reshape(-1), pred_train, pred_test, fig_path)
     
def cal_index(x):
    idx = np.where(x == 1)[0]
    if len(idx) != 2:
        raise ValueError('Compounds should exactly contains two.')
    return idx[0] * len(x) + idx[1]
    

###########################       main      ##############################
data_param = {'VAL':     'doseagg_min.cancer_gene', 
              'SCORE':   'RATIOADJ_SYN', 
              'FEATURE': ['comp'], 
              'CONCAT':  'CONCAT', 
              'RANDOM':  1234}
random_guess(X_train, X_test, y_train, y_test, model_name, 'median')

model_name, X_train, X_test, y_train, y_test = data_loader.read_data(**data_param)