############################################################################
###                            neural_net.py                             ###
############################################################################

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization, ReLU, Dropout, concatenate, multiply, add
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import keras.backend as K
import tensorflow as tf
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

def weighted_mse_v1(y_true, y_pred):
    loss = K.square(y_true - y_pred)
    weight_incremental = 2 * K.abs(y_true) + 1
    weight_flat = tf.ones(tf.shape(y_true))
    weight = tf.where(K.less_equal(K.abs(y_true), 0.5), weight_incremental, weight_flat)
    return K.mean(loss * weight)

def weighted_mse_v2(y_true, y_pred):
    loss = K.square(y_true - y_pred)
    weight_incremental = 4 * K.abs(y_true) + 1
    weight_flat = tf.ones(tf.shape(y_true))
    weight = tf.where(K.less_equal(K.abs(y_true), 0.25), weight_incremental, weight_flat)
    return K.mean(loss * weight)

def weighted_mse_v3(y_true, y_pred):
    loss = K.square(y_true - y_pred)
    weight_neg = 5 * K.abs(y_true) + 1
    weight_pos = tf.ones(tf.shape(y_true))
    weight = tf.where(K.less_equal(y_true, 0), weight_neg, weight_pos)
    return K.mean(loss * weight)

def weighted_mse_v4(y_true, y_pred):
    loss = K.square(y_true - y_pred)
    weight_large = K.exp(0.5 - y_true)
    weight_one = tf.ones(tf.shape(y_true))
    weight = tf.where(K.less_equal(y_true, 0.5), weight_large, weight_one)
    return K.mean(loss * weight)


##################################    model    ########################################
class neural_net(object):
    '''
    Neural network model
    '''
    def __init__(self, input_length, output_length=1, loss='mse', loss_weights=None, encoding_layer_sizes=None, fully_connected_layer_sizes=(128, 64, 32), merge='concat', dropout=0.2, learning_rate=1e-4, optimizer='Adam'):
        self.input_length                = input_length
        self.output_length               = output_length
        self.loss_weights                = loss_weights
        self.encoding_layer_sizes        = encoding_layer_sizes
        self.fully_connected_layer_sizes = fully_connected_layer_sizes
        self.merge                       = merge
        self.dropout                     = dropout
        self.learning_rate               = learning_rate
        self._parse_loss(loss)                  # loss, metrics, output_activation
        self._parse_optimizer(optimizer)        # optimizer
        self.model = self._model_initializer()


    def _model_initializer(self):
        print('Initializing connected neural network model ...', end='', flush=True)
        # fully connect network
        if self.encoding_layer_sizes is None and len(self.input_length) == 1:
            inputs = Input(shape=(self.input_length[0],), name='input')
            outputs = self._fully_connected_layers(inputs=inputs, layer_sizes=self.fully_connected_layer_sizes+(self.output_length,), dropout_rate=self.dropout, last_norm=False, output_activation=self.output_activation)
        # encoding input features than fully connect
        elif self.encoding_layer_sizes is not None and len(self.input_length) >= 2:
            if len(self.input_length) != len(self.encoding_layer_sizes):
                raise ValueError('Input number {} and encoding layer size {} should match.'.format(self.input_length, self.encoding_layer_sizes))
            inputs = [Input(shape=(length,), name='input'+str(i)) for i, length in enumerate(self.input_length)]
            features = [self._fully_connected_layers(inputs=inputs[i], layer_sizes=layers, dropout_rate=self.dropout, last_norm=True) for i, layers in enumerate(self.encoding_layer_sizes)]
            if self.merge == 'concat':
                features = concatenate(features)
            elif 'film' in self.merge:
                x, m = list(map(int, self.merge.split('-')[-1].split(':')))
                film = self._FiLM_layer(features[x], features[m])
                features = concatenate([film] + [f for i, f in enumerate(features) if i not in [x,m]])
            outputs = self._fully_connected_layers(inputs=features, layer_sizes=self.fully_connected_layer_sizes+(self.output_length,), dropout_rate=self.dropout, last_norm=False, output_activation=self.output_activation)
        else:
            raise ValueError('Input number {} and encoding layer size {} should match.'.format(self.input_length, self.encoding_layer_sizes))
        # construct
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.loss, loss_weights=self.loss_weights, optimizer=self.optimizer, metrics=self.metrics)
        print(' Done\nModel structure summary:', flush=True)
        print(model.summary())
        return model


    def train(self, X_train, y_train, model_name, model_path, validation_split=0.0, validation_data=None, batch_size=32, epochs=200, verbose=1, tolerance=10):
        print('Start training neural network ... ', end='', flush=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.tol = tolerance
        # path
        model_name = os.path.join(model_path, '{}@in_{}.out_{}.loss_{}.lw_{}.ecl_{}.fcl_{}.merge_{}.lr_{}.drop_{}.batch_{}.epoch_{}.h5')
        model_name = model_name.format(self.model_name, self.input_length, self.output_length, self.loss_name, self.loss_weights, self.encoding_layer_sizes, self.fully_connected_layer_sizes, self.merge,
                                       self.learning_rate, self.dropout, self.batch_size, self.epochs)
        model_name = model_name.replace(' ', '')
        log_dir = model_name.strip('.h5') + '.log'
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        # monitor
        early_stopper = EarlyStopping(patience=tolerance, verbose=verbose)
        check_pointer = ModelCheckpoint(model_name, verbose=verbose, save_best_only=True)
        # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-4, verbose=1)
        log = TensorBoard(log_dir=log_dir)
        result = self.model.fit(X_train, y_train, 
                                validation_split=validation_split, validation_data=validation_data,
                                batch_size=batch_size, epochs=epochs, verbose=verbose,shuffle=True,
                                callbacks=[early_stopper, check_pointer, log])
        self.loadModel(model_name)
        return result.history


    def loadModel(self, path):
        print('Loading trained neural network model ... ', end='', flush=True)
        if self.loss_name in ['mse', 'mae']:
            self.model = load_model(path, custom_objects={'r_square': r_square})
        elif 'weighted_mse' in self.loss_name:
            self.model = load_model(path, custom_objects={'r_square': r_square, self.loss_name: eval(self.loss_name)})
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
    

    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  utility  <<<<<<<<<<<<<<<<<<<<<<<<<<<<< ###
    def _parse_loss(self, loss):
        '''
        Parse loss function.
        '''
        if loss in ['mse', 'mae']:
            self.loss_name = self.loss = loss
            self.metrics = [r_square]
            self.output_activation = 'linear'
        elif 'weighted_mse' in loss:
            self.loss_name = loss
            self.loss = eval(loss)
            self.metrics = [r_square]
            self.output_activation = 'linear'
        elif loss == 'binary_crossentropy':
            self.loss_name = self.loss = loss
            self.metrics = ['accuracy']
            self.output_activation = 'sigmoid'
        elif loss == 'sparse_categorical_crossentropy':
            if self.output_length <= 2:
                raise ValueError('output_length should be larger than 2, if use sparse_categorical_crossentropy loss.')
            self.loss_name = self.loss = loss
            self.metrics = ['accuracy']
            self.output_activation = 'softmax'
        else:
            raise ValueError('Unrecognizable loss. Either mse or cross_entropy.')
    
    def _parse_optimizer(self, optimizer):
        '''
        Parse optimizer function.
        '''
        if optimizer == 'Adam':
            self.optimizer = Adam(lr=self.learning_rate)
        elif optimizer == 'RMSprop':
            self.optimizer = RMSprop(lr=self.learning_rate, decay=1e-6)
        else:
            raise ValueError('Unrecognizable optimizer. Either Adam or RMSprop.')
    
    @staticmethod
    def _fully_connected_layers(inputs, layer_sizes, dropout_rate, last_norm, output_activation=None):
        '''
        Fully connected layers
        '''
        if layer_sizes is None: return inputs
        for i, neurons in enumerate(layer_sizes):
            if i < len(layer_sizes) - 1:    # hidden layer
                dense = Dense(neurons) (inputs)
                bn = BatchNormalization() (dense)
                relu = ReLU() (bn)
                inputs = Dropout(dropout_rate) (relu)
            else:                           # last layer, decide if it is the final output
                if last_norm:               # normalize last layer
                    dense = Dense(neurons) (inputs)
                    bn = BatchNormalization() (dense)
                    outputs = ReLU() (bn)
                else:                       # output layer, not normalize
                    if output_activation not in ['linear', 'sigmoid', 'softmax']:
                        raise ValueError('Unrecognizable activation function {}: should be linear, sigmoid or softmax.'.format(output_activation))
                    if output_activation == 'softmax' and neurons <= 2:
                        raise ValueError('Output length should be larger than 2 when using softmax activation.')
                    outputs = Dense(neurons, activation=output_activation) (inputs) 
        return outputs
    
    @staticmethod
    def _FiLM_layer(x, modulate):
        x_shape = x.get_shape().as_list()[1]
        m_shape = modulate.get_shape().as_list()[1]
        if x_shape != m_shape: raise ValueError('x shape and modulate shape should be same for film mergeing.')
        gamma = Dense(m_shape, activation='sigmoid') (modulate)
        beta = Dense(m_shape, activation='tanh') (modulate)
        return add([multiply([x, gamma]), beta])


##################################    model    ########################################
# class dataModel(object):
#     def __init__(self, model, encoder, features, scaler):
#         self.model = model
#         self.encoder = encoder,
#         self.features = self.features
#         self.scaler = self.scaler

#     def dump(self, path):
#         pkl.dump(self, path)
