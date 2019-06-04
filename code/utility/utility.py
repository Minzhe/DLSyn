#############################################################################
###                               utility.py                              ###
#############################################################################
import re
import os
import pandas as pd
import pickle as pkl

### >>>>>>>>>>>>>>>>>>>>>>>>>>>>   clean string  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ###
def cleanCellName(name):
    '''
    Get alias of cell line name.
    '''
    alias = name.replace('/ATCC', '').replace(' ', '').upper()
    alias = re.sub(r'\(.*\)', '', alias)
    alias = re.sub(r'[-_\[\]]', '', alias)
    if alias == '7860':
        alias = '786O'
    elif alias == 'SR':
        alias = 'SR786'
    elif alias == 'NCI/ADRRES':
        alias = 'NCIADRRES'
    elif alias == 'MDAMB435':
        alias = 'MDAMB435S'
    elif alias == 'U251':
        alias = 'U251MG'
    elif alias == 'NCIH322M':
        alias = 'NCIH322'
    elif alias == 'NIHOVCAR3':
        alias = 'OVCAR3'
    return alias

def index_array(arr):
    index = dict()
    for i, x in enumerate(arr):
        index[i] = x
    return index


### >>>>>>>>>>>>>>>>>>>>>>>>>>>   read data   <<<<<<<<<<<<<<<<<<<<<<<<<<<<< ###
def read_cancer_gene():
    path = os.path.realpath(__file__)
    path = os.path.join('/'.join(path.split('/')[:-3]), 'data/Curated/cancer.gene.anno.csv')
    genes = pd.read_csv(path, usecols=['Gene'], squeeze=True)
    return genes

def read_compound_index():
    path = os.path.realpath(__file__)
    path = os.path.join('/'.join(path.split('/')[:-3]), 'data/Curated/ALMANAC/train_test_data/curated.combo.syn.doseagg_min.cancer_gene.ccle_norm.pkl')
    with open(path, 'rb') as f:
        data = pkl.load(f)
    index = data['comp_index']
    return {key: int(val.replace('COMP_', '')) for key, val in index.items()}