#############################################################################
###                               utility.py                              ###
#############################################################################
import numpy as np
import matplotlib as plt
import seaborn as sns
sns.set_style('white')
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

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
