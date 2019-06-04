##################################################################################
###                            predict.ALMANAC.DNN.py                          ###
##################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
# proj_dir = 'Z:/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
from model import neural_net, data_loader
import matplotlib.pyplot as plt
import utility.plotting as p
from predict_ALMANAC_DNN import dnn_model

##############################      function      ###############################
def param_search(param):
    data_param = {'VAL':     'doseagg_min.cancer_gene.ccle', 
                  'SCORE':   'RATIOADJ_SYN', 
                  'FEATURE': ['expr', 'mut', 'comp'], # ['panel', 'cell', 'comp']
                  'CONCAT':  'INDEP', 
                  'RANDOM':  1234,
                  'SPLIT':   'POINT'}
    model_name, X_train, X_test, y_train, y_test = data_loader.read_data(**data_param)
    dnn = dnn_model(model_name, X_train, X_test, y_train, y_test)
    res = dict()
    if param == 'batch_size':
        batch_size = [64, 128, 256, 512, 1028]
        for b in batch_size:
            dnn.batch_size = b
            res[b] = get_mse_r2(dnn)
    elif param == 'learning_rate':
        lr = [1e-4, 2e-4, 3e-4, 1e-3, 2e-3, 3e-3]
        for l in lr:
            dnn.learning_rate = l
            res[l] = get_mse_r2(dnn)
    elif param == 'drop_out':
        dropout = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
        for d in dropout:
            dnn.dropout = d
            res[d] = get_mse_r2(dnn)
    elif param == 'encoding':
        encoding_layer_sizes = [((512,256,256),(512,256,256),None,), ((512,256,128),(512,256,128),None,), ((256,256,128),(256,256,128),None,), ((256,256,128,128),(256,256,128,128),None,)]
        for el in encoding_layer_sizes:
            dnn.encoding_layer_sizes = el
            res[el] = get_mse_r2(dnn)
    elif param == 'connecting':
        connecting = [(256,256,256), (256,256,128,128), (256,256), (256,128,128,64)]
        for c in connecting:
            dnn.fully_connected_layer_sizes = c
            res[c] = get_mse_r2(dnn)
    res = pd.DataFrame(res).T
    # output
    out_path = os.path.join(proj_dir, 'result/ALMANAC.prediction/param_search.{}.csv'.format(param))
    res.to_csv(out_path)
            

def get_mse_r2(model):
    model.build()
    model.fit()
    model.load()
    return model.evaluate_train_val()

################################       main      #####################################
# param_search(param='batch_size')
# param_search(param='learning_rate')
# param_search(param='drop_out')
# param_search(param='encoding')
param_search(param='connecting')

