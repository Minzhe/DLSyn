#############################################################################
###                            curate.ALMANIC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
# proj_dir = 'Z:/bioinformatics/s418336/projects/DLSyn'
import argparse
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from model import neural_net
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.style.use('seaborn')

#############################    function   #############################
def train_cnn(val, col):
    data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.syn.{}_.pkl'.format(val))
    fig_path = os.path.join(proj_dir, 'result/ALMANAC.prediction/img/curated.combo.syn.{}.dnn.png'.format(val))
    model_path = os.path.join(proj_dir, 'code/model/archive')

    with open(data_path, 'rb') as f:
        data = pkl.load(f)
        X, y = data['in_array'], data['out_array'][col]
        X = X.loc[y.notnull(),:]
        y = y[y.notnull()]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    # x = 100000
    # X_train, X_test, y_train, y_test = X_train.iloc[:x,], X_test.iloc[:x,], y_train[:x], y_test[:x]

    # >>>>>>>>>>>>>>>>>>   model   <<<<<<<<<<<<<<<<<< #
    # model = GradientBoostingRegressor(n_estimators=200, verbose=1)
    # model.fit(X_train, y_train)
    model = neural_net.fcnn(input_length=X_train.shape[1], loss='mse', hidden_layer_sizes=(256,256,256), output_activation='linear', output_length=1, dropout=0.25, learning_rate=3e-4)
    model.train(X_train, y_train, model_name='{}-{}@'.format(col, val), model_path=model_path, validation_data=(X_test, y_test), epochs=200, batch_size=256, tolerance=30)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    print(r2_train, r2_test)




##############################    main   ###################################
parser = argparse.ArgumentParser(description='Call DNN models.')
parser.add_argument('-v', '--val', help='value to train model(mean, max or min).')
parser.add_argument('-s', '--score', help='output value (SCORE, RATIO_SYN or RATIOADJ_SYN).')
args = parser.parse_args()
if args.val not in ['mean', 'max', 'min'] and args.score not in ['SCORE', 'RATIO_SYN', 'RATIOADJ_SYN', 'SCORE_SYN']:
    raise ValueError('Parameter not recognizable.')
train_cnn(val=args.val, col=args.score)
exit()

# sns.distplot(y_test)
# plt.savefig(fig_path)


