#############################################################################
###                            curate.DrugSyn.py                          ###
#############################################################################

import os
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
# proj_dir = 'Z:/bioinformatics/s418336/projects/DLSyn'

##########################    function   ###########################
def prep_train_test(expr, syn, test_size=0.25):
    pairs, X, y = [], [], []
    for idx, row in syn.iterrows():
        pairs.append((syn['drugA'][idx], syn['drugB'][idx]))
        pairs.append((syn['drugB'][idx], syn['drugA'][idx]))
    for druga, drugb in pairs:
        X.append(list(expr[druga]) + list(expr[drugb]))
    return np.array(X)


##########################    main   ###########################
expr_path = os.path.join(proj_dir, 'data/DrugSyn/DrugSyn.Expr.csv')
gene_path = os.path.join(proj_dir, 'data/Curated/cancer.gene.anno.csv')
syn_path = os.path.join(proj_dir, 'data/DrugSyn/valid.vs.pred.csv')
fig_path = os.path.join(proj_dir, 'result/cluster.png')

gene = pd.read_csv(gene_path, usecols=['Gene'], squeeze=True)
expr = pd.read_csv(expr_path)
expr = expr.loc[expr['Genename'].isin(gene),:].groupby(by=['Genename']).max()
neg_control = expr[['Neg_control', 'Neg_control.1']].mean(axis=1)
expr = expr.apply(lambda x: x - neg_control)
syn = pd.read_csv(syn_path)
X = prep_train_test(expr, syn, 0.25)
print(X.shape)

# plt.subplots(figsize=(12,12))
# sns.clustermap(expr)
# plt.savefig(fig_path)