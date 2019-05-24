#############################################################################
###                            curate.ALMANIC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
# proj_dir = 'Z:/bioinformatics/s418336/projects/DLSyn'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from model import neural_net
import matplotlib.pyplot as plt
import utility.plotting as p

#############################    function   #############################
class dnn_model(object):
    def __init__(self, model_name, X_trian, X_test, y_train, y_test):
        # model param
        self.loss                        = 'weighted_mse_v4'
        self.loss_weights                = None
        self.encoding_layer_sizes        = ((512,256,256),(256,256,128),(128,128,64),None,)
        self.fully_connected_layer_sizes = (256,256,256)
        self.learning_rate               = 3e-4
        self.dropout                     = 0.25
        self.batch_size                  = 256
        self.epochs                      = 200
        self.tol                         = 30
        # path
        self.fig_path   = os.path.join(proj_dir, 'result/ALMANAC.prediction/img/{}.{}.png'.format(model_name, self.loss))
        self.model_path = os.path.join(proj_dir, 'code/model/archive')
        self.model_name = model_name
        # data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.input_length = (self.X_train.shape[1],) if not isinstance(self.X_train, list) else tuple(x.shape[1] for x in self.X_train)
        self.output_length = self.y_train.shape[1]
        # model      
        self.dnn = neural_net.neural_net(input_length=self.input_length, output_length=self.output_length, loss=self.loss, loss_weights=self.loss_weights,
                                         encoding_layer_sizes=self.encoding_layer_sizes, fully_connected_layer_sizes=self.fully_connected_layer_sizes, 
                                         dropout=self.dropout, learning_rate=self.learning_rate)
    
    def fit(self):
        self.dnn.train(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), 
                       model_name=self.model_name, model_path=self.model_path, 
                       epochs=self.epochs, batch_size=self.batch_size, tolerance=self.tol, verbose=2)

    def load(self):
        path = os.path.join(self.model_path, '{}@in_{}.out_{}.loss_{}.lw_{}.ecl_{}.fcl_{}.lr_{}.drop_{}.batch_{}.epoch_{}.tol_{}.h5')
        path = path.format(self.model_name, self.input_length, self.output_length, self.loss, self.loss_weights, self.encoding_layer_sizes, self.fully_connected_layer_sizes,
                           self.learning_rate, self.dropout, self.batch_size, self.epochs, self.tol)
        path = path.replace(' ', '')
        self.dnn.loadModel(path)

    def plot_train_val(self):
        pred_train = self.dnn.predict(self.X_train)
        pred_test = self.dnn.predict(self.X_test)
        f, ax = plt.subplots(2, 2, figsize=(16,16))
        p.plot_correlation(self.y_train, pred_train, 'Training set', ax[0][0])
        p.plot_correlation(self.y_test, pred_test, 'Validation set', ax[0][1])
        p.plot_density(self.y_train, ax[1][0])
        p.plot_density(self.y_test, ax[1][1])
        f.savefig(self.fig_path)
    
    def plot_train_val_subtype(self, split):
        cols = [col for col in sorted(list(set(self.X_train.columns) | set(self.X_test.columns)))]
        if split == 'drug':
            cols = [col for col in cols if 'COMP' in col]
            fig_dir = os.path.join(os.path.dirname(self.fig_path), os.path.basename(self.fig_path).strip('png') + 'compound')
        elif split == 'cell':
            cols = [col for col in cols if 'COMP' not in col and 'Cancer' not in col]
            fig_dir = os.path.join(os.path.dirname(self.fig_path), os.path.basename(self.fig_path).strip('png') + 'cell')
        elif split == 'type':
            cols = [col for col in cols if 'Cancer' in col]
            fig_dir = os.path.join(os.path.dirname(self.fig_path), os.path.basename(self.fig_path).strip('png') + 'type')
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        for col in cols:
            train_idx = self.X_train[col] == 1
            test_idx = self.X_test[col] == 1
            X_train, y_train = self.X_train.loc[train_idx,:], self.y_train.loc[train_idx]
            X_test, y_test = self.X_test.loc[test_idx,:], self.y_test.loc[test_idx]
            pred_train = self.dnn.predict(X_train)
            pred_test = self.dnn.predict(X_test)
            f, ax = plt.subplots(1, 2, figsize=(16,8))
            p.plot_correlation(y_train, pred_train, 'Training set', ax[0])
            p.plot_correlation(y_test, pred_test, 'Validation set', ax[1])
            f.suptitle(col)
            path = os.path.join(fig_dir, os.path.basename(self.fig_path).strip('png') + '{}.png'.format(col.replace('/', '')))
            f.savefig(path)
            plt.close()


# >>>>>>>>>>>>>>>>>>>>>>>>>     data loader    <<<<<<<<<<<<<<<<<<<<<<<<<<< #
def read_data(VAL, SCORE, FEATURE, CONCAT, RANDOM):
    '''
    Read training and testing data.
    '''
    data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/train_test_data/curated.combo.syn.{}.pkl'.format(VAL))
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    X = [data[f+'_array'] for f in FEATURE]
    y = data['output_array'][SCORE].values.reshape((-1,1))
    if CONCAT == 'CONCAT':
        X = np.concatenate(X, axis=1)
    # remove nan
    if isinstance(X, list):
        nanidx = np.array([np.apply_along_axis(any, 1, np.isnan(x)) for x in X])
        nanidx = np.apply_along_axis(any, 0, nanidx)
        X = [x[~nanidx,:] for x in X]
        y = y[~nanidx,:]
    else:
        nanidx = np.apply_along_axis(any, 1, np.isnan(X))
        X = X[~nanidx,:]
        y = y[~nanidx,:]
    # split data
    np.random.seed(RANDOM)
    idx = np.random.choice(y.shape[0], int(0.2*y.shape[0]), replace=False)
    idx = np.isin(list(range(y.shape[0])), idx)
    y_train, y_test = y[~idx,:], y[idx,:]
    if isinstance(X, list):
        X_train = [x[~idx,:] for x in X]
        X_test = [x[idx,:] for x in X]
    else:
        X_train, X_test = X[~idx,:], X[idx,:]
    # name
    model_name = '{}-{}-{}-{}'.format(VAL.upper(), SCORE, '_'.join([x.upper() for x in FEATURE]), CONCAT)
    return model_name, X_train, X_test, y_train, y_test



##############################    main   ###################################
data_param = {'VAL':     'doseagg_min.cancer_gene', 
              'SCORE':   'RATIOADJ_SYN', 
              'FEATURE': ['expr', 'mut', 'protein', 'comp'], # ['panel', 'cell', 'comp']
              'CONCAT':  'INDEP', 
              'RANDOM':  1234}
model_name, X_train, X_test, y_train, y_test = read_data(**data_param)

dnn = dnn_model(model_name, X_train, X_test, y_train, y_test)
dnn.fit()
dnn.load()
dnn.plot_train_val()
# dnn.plot_train_val_subtype(split='drug')