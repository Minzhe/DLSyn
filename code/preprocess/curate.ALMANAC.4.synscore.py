#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import utility.utility as util

###############################      function     ##############################
def cal_syn_score(single_path, combo_path, out_path, method):
    single = pd.read_csv(single_path)
    combo = pd.read_csv(combo_path)
    # get single effect score
    combo = append_single(combo, single)
    print('Calculating synergy score ...')
    combo['RATIO_EXPECTED'] = get_expected(combo['RATIO1'], combo['RATIO2'], method=method)
    combo['RATIOADJ_EXPECTED'] = get_expected(combo['RATIOADJ1'], combo['RATIOADJ2'], method=method)
    combo['RATIO_SYN'] = combo['RATIO'] - combo['RATIO_EXPECTED']
    combo['RATIOADJ_SYN'] = combo['RATIOADJ'] - combo['RATIOADJ_EXPECTED']
    combo['SCORE_SYN'] = combo['RATIOADJ'] - combo['EXPECTED']
    combo['CELL'] = combo['CELL'].apply(util.cleanCellName)
    combo.to_csv(out_path, index=None)

def cal_min_syn_score(single_path, combo_path, out_path, method):
    single = pd.read_csv(single_path)
    combo = pd.read_csv(combo_path)
    # get single effect score
    combo = append_single(combo, single, conc=False)
    print('Calculating synergy score ...')
    combo['RATIO_EXPECTED'] = get_expected(combo['RATIO1'], combo['RATIO2'], method=method)
    combo['RATIOADJ_EXPECTED'] = get_expected(combo['RATIOADJ1'], combo['RATIOADJ2'], method=method)
    combo['RATIO_SYN'] = combo['RATIO'] - combo['RATIO_EXPECTED']
    combo['RATIOADJ_SYN'] = combo['RATIOADJ'] - combo['RATIOADJ_EXPECTED']
    combo['SCORE_SYN'] = combo['RATIOADJ'] - combo['EXPECTED']
    combo['CELL'] = combo['CELL'].apply(util.cleanCellName)
    out_path = out_path.strip('.csv') + '.{}.csv'.format(method)
    combo.to_csv(out_path, index=None)


### >>>>>>>>>>>>>>>>>>>>>> utility <<<<<<<<<<<<<<<<<<<<<< ###
def append_single(combo, single, conc=True):
    single = create_single_dict(single, conc)
    if conc:
        first = ['TYPE', 'CELL', 'COMP1', 'CONC1']
        second = ['TYPE', 'CELL', 'COMP2', 'CONC2']
    else:
        first = ['TYPE', 'CELL', 'COMP1']
        second = ['TYPE', 'CELL', 'COMP2']
    print('Retriving first compound ...')
    combo = combo.groupby(by=first, as_index=False, sort=False).apply(lambda x: get_single(x, first, '1', single))
    print('Retriving second compound ...')
    combo = combo.groupby(by=second, as_index=False, sort=False).apply(lambda x: get_single(x, second, '2', single))
    return combo

def get_single(df, col, suffix, single):
    info = df[col].iloc[0,:].squeeze()
    ratio, ratio_adj = single.get(tuple(info), (np.nan, np.nan))
    df['RATIO'+suffix] = ratio
    df['RATIOADJ'+suffix] = ratio_adj
    return df

def create_single_dict(df, conc):
    if conc:
        eff = {(row['TYPE'], row['CELL'], row['COMP1'], row['CONC1']): (row['RATIO'], row['RATIOADJ']) for idx, row in df.iterrows()}
    else:
        eff = {(row['TYPE'], row['CELL'], row['COMP1']): (row['RATIO'], row['RATIOADJ']) for idx, row in df.iterrows()}
    return eff

def get_expected(x1, x2, method):
    score = np.array([np.nan] * len(x1))
    if method == 'nci':
        idx = (x1 <= 0) | (x2 <= 0)
        score[idx] = np.minimum(x1, x2)[idx]
        score[~idx] = (np.minimum(x1, 1) * np.minimum(x2, 1))[~idx]
    elif method == 'hsa':
        score = np.minimum(x1, x2)
    return score


#################################    main   ###################################
combo_path  = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.score.csv')
single_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.single.score.csv')
out_path    = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.csv')
combo_min_path  = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.score.doseagg_min.csv')
single_min_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.single.score.doseagg_min.csv')
out_min_path    = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.doseagg_min.csv')

# cal_syn_score(single_path, combo_path, out_path)
cal_min_syn_score(single_min_path, combo_min_path, out_min_path, method='hsa')
