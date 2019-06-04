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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from model import neural_net, data_loader
import matplotlib.pyplot as plt
import utility.plotting as p


#############################    function   #############################
class dnn_model(object):
    def __init__(self, model_name, X_train, X_test, y_train, y_test):
        # model param
        self.loss                        = 'weighted_mse_v4'
        self.loss_weights                = None
        self.encoding_layer_sizes        = ((256,256,128),(256,256,128),None,)
        self.fully_connected_layer_sizes = (256,256,256)
        self.merge                       = 'concat'
        self.learning_rate               = 1e-3
        self.dropout                     = 0.1
        self.batch_size                  = 128
        self.epochs                      = 250
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
    
    def build(self):
        # model      
        self.dnn = neural_net.neural_net(input_length=self.input_length, output_length=self.output_length, loss=self.loss, loss_weights=self.loss_weights,
                                         encoding_layer_sizes=self.encoding_layer_sizes, fully_connected_layer_sizes=self.fully_connected_layer_sizes, merge=self.merge,
                                         dropout=self.dropout, learning_rate=self.learning_rate)
    
    def fit(self):
        self.dnn.train(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), 
                       model_name=self.model_name, model_path=self.model_path, 
                       epochs=self.epochs, batch_size=self.batch_size, tolerance=self.tol, verbose=1)

    def load(self):
        path = os.path.join(self.model_path, '{}@in_{}.out_{}.loss_{}.lw_{}.ecl_{}.fcl_{}.merge_{}.lr_{}.drop_{}.batch_{}.epoch_{}.h5')
        path = path.format(self.model_name, self.input_length, self.output_length, self.loss, self.loss_weights, self.encoding_layer_sizes, self.fully_connected_layer_sizes, self.merge,
                           self.learning_rate, self.dropout, self.batch_size, self.epochs)
        path = path.replace(' ', '')
        self.dnn.loadModel(path)
    
    def predict(self, X):
        return self.dnn.predict(X)

    def plot_train_val(self):
        pred_train = self.dnn.predict(self.X_train)
        pred_test = self.dnn.predict(self.X_test)
        p.plot_correlation_train_val(self.y_train, self.y_test, pred_train, pred_test, self.fig_path)
    
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
    
    def evaluate_train_val(self):
        pred_train = self.dnn.predict(self.X_train)
        pred_test = self.dnn.predict(self.X_test)
        val = self.evaluate(self.y_train, pred_train) + self.evaluate(self.y_test, pred_test)
        index = ['train_r2', 'train_pearsonr', 'train_spearmanr', 'train_mse', 'test_r2', 'test_pearsonr', 'test_spearmanr', 'test_mse']
        return pd.Series(val, index=index)
 
    @staticmethod
    def evaluate(truth, pred):
        truth = np.reshape(truth, -1)
        pred = np.reshape(pred, -1)
        r2 = round(r2_score(truth, pred), 4)
        pearson_r = round(pearsonr(truth, pred)[0], 4)
        spearman_r = round(spearmanr(truth, pred)[0], 4)
        mse = round(mean_squared_error(truth, pred), 4)
        return r2, pearson_r, spearman_r, mse



##############################    main   ###################################
if __name__ == "__main__":
    mode = 'predict'
    data_param = {'VAL':     'doseagg_min.cancer_gene.ccle_norm', 
                  'SCORE':   'RATIOADJ_SYN', 
                  'FEATURE': ['expr', 'mut', 'comp'], # ['panel', 'cell', 'comp']
                  'CONCAT':  'INDEP', 
                  'RANDOM':  1234,
                  'SPLIT':   'POINT'}
    if mode == 'train':
        model_name, X_train, X_test, y_train, y_test = data_loader.read_data_nci60(**data_param)
        dnn = dnn_model(model_name, X_train, X_test, y_train, y_test)
        dnn.build()
        dnn.fit()
        dnn.load()
        dnn.plot_train_val()
        # dnn.plot_train_val_subtype(split='drug')
    elif mode == 'predict':
        # model
        model_name, X_train, X_test, y_train, y_test = data_loader.read_data_nci60(**data_param)
        dnn = dnn_model(model_name, X_train, X_test, y_train, y_test)
        dnn.build()
        dnn.load()
        # new data
        res_blank = data_loader.prepare_data_ccle(data_param['VAL'], data_param['FEATURE'])
        cells = np.unique(res_blank['CELL'])
        for cell in cells:
            out_path = os.path.join(proj_dir, 'result/ALMANAC.prediction/CCLE.prediction/Cell/{}.csv'.format(cell))
            if os.path.isfile(out_path) or cell == 'TT': continue
            print('* Predicting cell {} ...'.format(cell))
            tmp_res = res_blank.loc[res_blank['CELL'] == cell,:]
            X = data_loader.read_data_ccle(tmp_res, data_param['VAL'], data_param['FEATURE'], data_param['CONCAT'], cell)
            y = dnn.predict(X)
            tmp_res['SYN'] = y
            tmp_res.to_csv(out_path, index=None)