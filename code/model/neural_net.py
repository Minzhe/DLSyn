############################################################################
###                            neural_net.py                             ###
############################################################################

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization, ReLU, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as K
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import pickle as pkl

##################################    function    ########################################
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

def pearson_coef(y_true, y_pred):
    y_true_c = y_true - K.mean(y_true)
    y_pred_c = y_pred - K.mean(y_pred)
    return K.sum(y_true_c * y_pred_c) / (K.sqrt(K.sum(K.square(y_true_c)) * K.sum(K.square(y_pred_c))) + K.epsilon())

##################################    model    ########################################
class fcnn(object):
    '''
    Fully connected nerual network
    '''
    def __init__(self, input_length, loss, hidden_layer_sizes=(128, 64, 32), output_activation='linear', output_length=1, dropout=0.2, learning_rate=1e-4, optimizer='Adam'):
        self.input_length = input_length
        self.output_length = output_length
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.output_activation = output_activation
        if loss == 'mse':
            self.loss = 'mse'
            self.metrics = [r_square]
        elif loss == 'binary_crossentropy':
            self.loss = 'binary_crossentropy'
            self.metrics = ['accuracy']
        elif loss == 'sparse_categorical_crossentropy':
            if self.output_length <= 2:
                raise ValueError('output_length should be larger than 2, if use sparse_categorical_crossentropy loss.')
            self.loss = 'sparse_categorical_crossentropy'
            self.metrics = ['accuracy']
        else:
            raise ValueError('Unrecognizable loss. Either mse or cross_entropy.')
        if optimizer == 'Adam':
            self.optimizer = Adam(lr=self.learning_rate)
        elif optimizer == 'RMSprop':
            self.optimizer = RMSprop(lr=self.learning_rate, decay=1e-6)
        else:
            raise ValueError('Unrecognizable optimizer. Either Adam or RMSprop.')
        self.model = self._model_init()
    

    def _model_init(self):
        print('Initilizing fully connected nerual netowrk model ...', end='', flush=True)
        inputs = Input(shape=(self.input_length,), name='input')
        # hidden layers
        for i, neurons in enumerate(self.hidden_layer_sizes):
            if i == 0:
                dense = Dense(neurons) (inputs)
                bn = BatchNormalization() (dense)
                relu = ReLU() (bn)
                dropout = Dropout(self.dropout) (relu)
            else:
                dense = Dense(neurons) (dropout)
                bn = BatchNormalization() (dense)
                relu = ReLU() (bn)
                dropout = Dropout(self.dropout) (relu)
        if self.output_activation == 'linear':
            outputs = Dense(1, name='output') (dropout)
        elif self.output_activation == 'sigmoid':
            outputs = Dense(1, activation='sigmoid', name='output') (dropout)
        elif self.output_activation == 'softmax':
            outputs = Dense(self.output_length, activation='softmax', name='output') (dropout)
        else:
            raise ValueError('Unrecognizabel activation function {}: should be linear, sigmoid or softmax.'.format(self.output_activation))
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        print(' Done\nModel structure summary:', flush=True)
        print(model.summary())

        return model


    def train(self, X_train, y_train, model_name, model_path, validation_split=0.0, validation_data=None, batch_size=32, epochs=200, verbose=1, tolerance=10):
        print('Start training neural network ... ', end='', flush=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epchos = epochs
        self.tol = tolerance
        model_name = os.path.join(model_path, '{}.lr-{}.layers-{}.drop-{}.batch-{}.acti-{}.epcho-{}.tol-{}.h5'.format(self.model_name, self.learning_rate, self.hidden_layer_sizes, self.dropout, self.batch_size, self.output_activation, self.epchos, self.tol))
        log_dir = os.path.join(model_path, '{}.lr-{}.layers-{}.drop-{}.batch-{}.acti-{}.epcho-{}.tol-{}.log'.format(self.model_name, self.learning_rate, self.hidden_layer_sizes, self.dropout, self.batch_size, self.output_activation, self.epchos, self.tol))
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        # monitor
        early_stopper = EarlyStopping(patience=tolerance, verbose=verbose)
        check_pointer = ModelCheckpoint(model_name, verbose=verbose, save_best_only=True)
        log = TensorBoard(log_dir=log_dir)
        result = self.model.fit(X_train, y_train, 
                                validation_split=validation_split, 
                                validation_data=validation_data,
                                batch_size=batch_size, 
                                epochs=epochs, 
                                verbose=verbose,
                                shuffle=True,
                                callbacks=[early_stopper, check_pointer, log])
        if self.loss == 'mse':
            self.model = load_model(model_name, custom_objects={'r_square': r_square})
        else:
            self.model = load_model(model_name)
        print('Done')
        return result.history


    def loadModel(self, path):
        print('Loading trained neural network model ... ', end='', flush=True)
        if self.loss == 'mse':
            self.model = load_model(path, custom_objects={'r_square': r_square})
        else:
            self.model = load_model(path)
        print('Done')


    def predict(self, X, verbose=1, label=None):
        y_pred = self.model.predict(X, verbose=verbose)
        if label:
            if self.loss in ['binary_crossentropy', 'sparse_categorical_crossentropy']:
                y_pred = y_pred.argmax(axis=-1)
            else:
                raise ValueError('Predict classes only available in classification problem.')
        return y_pred
    

    def evaluate(self, X, y_true, metrics, **kwargs):
        y_pred = self.predict(X)
        if metrics == 'regression':
            res = pearsonr(y_true.reshape(-1), y_pred.reshape(-1))[0]
        elif metrics == 'classification':
            if self.loss == 'sparse_categorical_crossentropy':
                res = self.multi_class_metrics(y_true, y_pred, **kwargs)
            else:
                res = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
        else:
            raise ValueError('Unrecognizable metrics {}: only corr or auc.'.format(metrics))
        return res
    

    def evaluate_all(self, Xs, y_trues, suffix, label_encoder):
        df = list()
        for X, y_true, suf in zip(Xs, y_trues, suffix):
            df.append(self.evaluate(X, y_true, metrics='classification', label_encoder=label_encoder, suffix=suf))
        df = pd.concat(df, axis=1, sort=False)
        return df
    

    def multi_class_metrics(self, y_true, y_pred, label_encoder, suffix=None):
        y_pred_label = y_pred.argmax(axis=-1)
        # table
        labels = list(np.unique(label_encoder.inverse_transform(y_true)))
        metrics_table = pd.DataFrame(np.nan, index=labels+['All'], columns=['Acc', 'Count', 'Prec', 'Recall', 'Auc'])
        metrics_table.loc['All','Acc'] = round(accuracy_score(y_true=y_true, y_pred=y_pred_label), 3)
        metrics_table.loc['All','Count'] = len(y_true)
        # each labels
        for label in labels:
            idx = label_encoder.transform([label])[0]
            tmp_true = (y_true == idx).astype(int)
            tmp_pred = (y_pred_label == idx).astype(int)
            metrics_table.loc[label,'Acc'] = round(accuracy_score(y_true=tmp_true, y_pred=tmp_pred), 3)
            metrics_table.loc[label,'Count'] = np.sum(y_true == idx)
            metrics_table.loc[label,'Prec'] = round(precision_score(y_true=tmp_true, y_pred=tmp_pred), 3)
            metrics_table.loc[label,'Recall'] = round(recall_score(y_true=tmp_true, y_pred=tmp_pred), 3)
            metrics_table.loc[label,'Auc'] = round(roc_auc_score(y_true=tmp_true, y_score=y_pred[:,idx]), 3)
        if suffix is not None:
            metrics_table = metrics_table.add_suffix('_'+suffix)
            # metrics_table.sort_index(axis=1, inplace=True)
        return metrics_table


##################################    model    ########################################
# class dataModel(object):
#     def __init__(self, model, encoder, features, scaler):
#         self.model = model
#         self.encoder = encoder,
#         self.features = self.features
#         self.scaler = self.scaler

#     def dump(self, path):
#         pkl.dump(self, path)