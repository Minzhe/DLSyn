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
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from model import neural_net
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.style.use('seaborn')



##############################    main   ###################################
data_path = os.path.join(proj_dir, 'data/Curated/NCI.ALMANIC.double.swap.pkl')
fig_path = os.path.join(proj_dir, 'data/Curated/NCI.ALMANIC.double.swap.diff_adj_.png')
model_path = os.path.join(proj_dir, 'code/model/archive')

with open(data_path, 'rb') as f:
    data = pkl.load(f)
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'].iloc[:,2], data['y_test'].iloc[:,2]
# x = 100000
# X_train, X_test, y_train, y_test = X_train.iloc[:x,], X_test.iloc[:x,], y_train[:x], y_test[:x]

# sns.distplot(y_test)
# plt.savefig(fig_path)

# model = GradientBoostingRegressor(n_estimators=200, verbose=1)
# model.fit(X_train, y_train)
model = neural_net.fcnn(input_length=X_train.shape[1], loss='mse', hidden_layer_sizes=(256,256,256), output_activation='linear', output_length=1, dropout=0.25, learning_rate=3e-4)
model.train(X_train, y_train, model_name='test-all-diff_adj@', model_path=model_path, validation_data=(X_test, y_test), epochs=200, batch_size=256, tolerance=30)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
print(r2_train, r2_test)
