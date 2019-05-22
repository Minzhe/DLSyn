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
    def __init__(self, val, score, predict):
        self.loss                        = 'weighted_mse_v4'
        self.learning_rate               = 3e-4
        self.encoding_layer_sizes        = None
        self.fully_connected_layer_sizes = (256,256,256)
        self.dropout                     = 0.25
        self.batch_size                  = 256
        self.epochs                      = 200
        self.tol                         = 30
        self.data_path  = os.path.join(proj_dir, 'data/Curated/ALMANAC/train_test_data/curated.combo.syn.{}.pkl'.format(val))
        self.fig_path   = os.path.join(proj_dir, 'result/ALMANAC.prediction/img/curated.combo.syn.{}.dnn.lose_{}.{}.png'.format(val, self.loss, predict))
        self.model_path = os.path.join(proj_dir, 'code/model/archive')
        self.model_name = '{}-{}-{}'.format(score, val, predict)
        self.read_data()

                    
        self.dnn = neural_net.neural_net(input_length=self.X_train.shape[1], output_length=self.output_length, loss=self.loss, loss_weights=self.loss_weights,
                                         encoding_layer_sizes=self.encoding_layer_sizes, fully_connected_layer_sizes=self.fully_connected_layer_sizes, 
                                         dropout=self.dropout, learning_rate=self.learning_rate)
    
    def fit(self):
        self.dnn.train(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), 
                       model_name=self.model_name, model_path=self.model_path, 
                       epochs=self.epochs, batch_size=self.batch_size, tolerance=self.tol, verbose=1)

    def load(self):
        path = os.path.join(self.model_path, '{}@inlin_{}.outlen_{}.loss_{}.lossweight_{}.ecl_{}.fcl_{}.lr_{}.drop_{}.batch_{}.epoch_{}.tol_{}.h5')
        path = path.format(self.model_name,)
        path = os.path.join(self.model_path, '{}@outlen-{}.loss-{}.lossweight-{}.lr-{}.layers-{}.drop-{}.batch-{}.epcho-{}.tol-{}.h5'.format( 
                                                                                                                                             self.output_length,
                                                                                                                                             self.loss, 
                                                                                                                                             self.loss_weights,
                                                                                                                                             self.learning_rate,
                                                                                                                                             self.hidden_layer_sizes, 
                                                                                                                                             self.dropout, 
                                                                                                                                             self.batch_size, 
                                                                                                                                             self.epochs, 
                                                                                                                                             self.tol))
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
    
    # >>>>>>>>>>>>>>>>>>>>>>  utility function  <<<<<<<<<<<<<<<<<<<<<< #
    def read_data(self):
        '''
        Read training and testing data.
        '''
        with open(self.data_path, 'rb') as f:
            data = pkl.load(f)
            if val == 'min_max_mean':
                X, y = data['in_array'], data['out_array'][[col for col in data['out_array'].columns if score in col]]
                y = y.dropna()
                X = X.loc[y.index,:]
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
                self.y_train = [self.y_train[col] for col in self.y_train.columns]
                self.y_test = [self.y_test[col] for col in self.y_test.columns]
                self.output_length = len(self.y_train)
                self.loss_weights = [0.25,0.25,0.5]
            else:
                X, y = data['in_array'], data['out_array'][score]
                y = y.dropna()
                X = X.loc[y.index,:]
                X, y = self.subset_data(X, y, value=predict) if predict in ['large_value', 'small_value'] else (X, y)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
                self.output_length = 1
                self.loss_weights = None
                if self.loss == 'binary_crossentropy':
                    if predict != 'class':
                        raise ValueError('Loss and predicted value not match.')
                    self.y_train[abs(self.y_train) < 0.08] = 0
                    self.y_train[abs(self.y_train) >= 0.08] = 1
                    self.y_test[abs(self.y_test) < 0.08] = 0
                    self.y_test[abs(self.y_test) >= 0.08] = 1

    @staticmethod
    def subset_data(X, y, value):
        if value == 'small_value':
            y = y[abs(y) < 0.05]
            X = X.loc[y.index,:]
        elif value == 'large_value':
            y = y[abs(y) >= 0.05]
            X = X.loc[y.index,:]
        return X, y


##############################    main   ###################################
val     = 'doseagg_min'
score   = 'RATIOADJ_SYN'
mode    = 'load'
predict = 'value'
    
dnn = dnn_model(val=val, score=score, predict=predict)
if mode == 'train':
    dnn.fit()
elif mode == 'load':
    dnn.load()
dnn.plot_train_val()
# dnn.plot_train_val_subtype(split='drug')
