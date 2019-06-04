############################################################################
###                           data_loader.py                             ###
############################################################################
proj_dir = proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import numpy as np
import pandas as pd
import pickle as pkl
from itertools import combinations

# >>>>>>>>>>>>>>>>>>>>>>>>>     data loader    <<<<<<<<<<<<<<<<<<<<<<<<<<< #
def read_data_nci60(VAL, SCORE, FEATURE, CONCAT, RANDOM, SPLIT):
    '''
    Read training and testing data.
    '''
    data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/train_test_data/curated.combo.syn.{}.pkl'.format(VAL))
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    X = [data[f+'_array'] for f in FEATURE]
    y = data['output_array'][SCORE].values.reshape((-1,1))
    if CONCAT == 'CONCAT':
        X = np.concatenate(X, axis=1)
    # remove nan
    if isinstance(X, list):
        nanidx = np.array([np.apply_along_axis(any, 1, np.isnan(x)) for x in X])
        nanidx = np.apply_along_axis(any, 0, nanidx)
        X = [x[~nanidx,:] for x in X]
        y = y[~nanidx,:]
    else:
        nanidx = np.apply_along_axis(any, 1, np.isnan(X))
        X = X[~nanidx,:]
        y = y[~nanidx,:]
    # split data
    np.random.seed(RANDOM)
    if SPLIT == 'POINT':
        idx = np.random.choice(y.shape[0], int(0.2*y.shape[0]), replace=False)
        idx = np.isin(list(range(y.shape[0])), idx)
    elif SPLIT == 'CELL':
        cell_array = data['cell_array'][~nanidx,:]
        cell = list(range(cell_array.shape[1]))
        idx = np.random.choice(len(cell), int(0.2*len(cell)), replace=False)
        idx = np.apply_along_axis(lambda x: np.where(x == 1)[0] in idx, 1, cell_array)
    y_train, y_test = y[~idx,:], y[idx,:]
    if isinstance(X, list):
        X_train = [x[~idx,:] for x in X]
        X_test = [x[idx,:] for x in X]
    else:
        X_train, X_test = X[~idx,:], X[idx,:]
    # name
    model_name = '{}-{}-{}-{}-{}'.format(VAL.upper(), SCORE, '_'.join([x.upper() for x in FEATURE]), CONCAT, SPLIT)
    return model_name, X_train, X_test, y_train, y_test

########################################    prepare for new cell line data   ########################################
def prepare_data_ccle(VAL, FEATURE):
    '''
    Prepare blank ccle data information to predict.
    '''
    out_path = os.path.join(proj_dir, 'result/ALMANAC.prediction/CCLE.prediction/CCLE.{}.{}.pred_blank.csv'.format(VAL, '-'.join(FEATURE)))
    if os.path.isfile(out_path):
        print('Reading {} ...'.format(out_path))
        res = pd.read_csv(out_path)
        return res
    else:
        # read data
        data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/train_test_data/curated.combo.syn.{}.pkl'.format(VAL))
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        # cells
        X = dict()
        if 'expr' in FEATURE:
            expr_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_expr.cancer_gene.csv')
            X['expr'] = pd.read_csv(expr_path, index_col=0)
        if 'mut' in FEATURE:
            mut_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_mutation.cancer_gene.csv')
            X['mut'] = pd.read_csv(mut_path, index_col=0)
        cells = [set(val.index) for key, val in X.items()]
        cells = sorted(set.intersection(*cells))
        # compound
        comps = list(range(len(data['comp_index'])))
        comps = list(combinations(comps, 2))
        # res
        res = pd.DataFrame([(cell,) + comp for cell in cells for comp in comps], columns=['CELL', 'COMP1', 'COMP2'])
        res['SYN'] = np.nan
        print('Writing {} ...'.format(out_path))
        res.to_csv(out_path, index=None)
        return res

def read_cellline_genomic(path, cell, nrows, scaler, index):
    '''
    Read cell line genomic information.
    '''
    data = pd.read_csv(path, index_col=0)
    data = data.loc[data.index == cell,:]
    # check column consistency
    if list(data.columns) != list(index.values()):
        raise ValueError('Input columns does not match training data columns.')
    # scale
    if scaler is not None:
        data[data.columns] = scaler.transform(data)
    data = pd.concat([data] * nrows, ignore_index=True).values
    return data

def read_comp_combination(DF, index):
    '''
    Generate compound combination array.
    '''
    def array_one(n, x):
        arr = np.zeros(n)
        arr[x] = 1
        return arr
    n = len(index)
    comp = np.apply_along_axis(lambda x: array_one(n, x), 1, DF.values)
    return comp

def read_data_ccle(DF, VAL, FEATURE, CONCAT, CELL):
    '''
    Prepare input array of CCLE cell line for neural network.
    '''
    # read data
    data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/train_test_data/curated.combo.syn.{}.pkl'.format(VAL))
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    # feature
    X = list()
    for f in FEATURE:
        print('== Preparing {} input ...'.format(f))
        if f in ['mut', 'expr', 'meta']:
            path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_{}.cancer_gene.csv'.format(f))
            scaler = None if f == 'mut' else data[f+'_scaler']
            X.append(read_cellline_genomic(path, CELL, DF.shape[0], scaler, data[f+'_index']))
        elif f == 'comp':
            X.append(read_comp_combination(DF[['COMP1', 'COMP2']], data[f+'_index']))
        else:
            raise ValueError('Unrecognized data type.')
    if CONCAT == 'CONCAT':
        X = np.concatenate(X, axis=1)
    return X