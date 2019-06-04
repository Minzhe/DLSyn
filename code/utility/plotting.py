#############################################################################
###                               utility.py                              ###
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

#########################       plot correlation      ##########################
def plot_correlation_train_val(y_train, y_test, pred_train, pred_test, fig_path):
    f, ax = plt.subplots(2, 2, figsize=(16,16))
    plot_correlation(y_train, pred_train, 'Training set', ax[0][0])
    plot_correlation(y_test, pred_test, 'Validation set', ax[0][1])
    plot_density(y_train, ax[1][0])
    plot_density(y_test, ax[1][1])
    f.savefig(fig_path)

def plot_correlation(truth, pred, title, ax):
    truth = np.reshape(truth, -1)
    pred = np.reshape(pred, -1)
    lims = min(min(pred), min(truth)), max(max(pred), max(truth))
    r2 = round(r2_score(truth, pred), 4)
    pearson_r = round(pearsonr(truth, pred)[0], 4)
    spearman_r = round(spearmanr(truth, pred)[0], 4)
    mse = round(mean_squared_error(truth, pred), 4)
    mae = round(mean_absolute_error(truth, pred), 4)
    ax.scatter(truth, pred, s=5)
    ax.set_title(title)
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    ax.text(lims[0], lims[1], s='r2_score: {}\npearsonr: {}\nspearmanr: {}\nmse: {} \nmae: {}'.format(str(r2), str(pearson_r), str(spearman_r), str(mse), str(mae)), verticalalignment='top')
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

def plot_density(x, ax):
    sns.distplot(x, ax=ax)
