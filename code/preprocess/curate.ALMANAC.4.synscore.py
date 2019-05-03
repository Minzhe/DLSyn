#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np

###############################      function     ##############################
def cal_syn_score(single_path, combo_path):
    single = pd.read_csv(single_path)
    combo = pd.read_csv(combo_path)
    # get single effect score
    combo = append_single(combo, single)
    print('Calculating synergy score ...')
    combo['RATIO_EXPECTED'] = get_expected(combo['RATIO1'], combo['RATIO2'])
    combo['RATIOADJ_EXPECTED'] = get_expected(combo['RATIOADJ1'], combo['RATIOADJ2'])
    combo['RATIO_SYN'] = combo['RATIO'] - combo['RATIO_EXPECTED']
    combo['RATIOADJ_SYN'] = combo['RATIOADJ'] - combo['RATIOADJ_EXPECTED']
    combo['SCORE_SYN'] = combo['RATIOADJ'] - combo['EXPECTED']
    return combo

### >>>>>>>>>>>>>>>>>>>>>> utility <<<<<<<<<<<<<<<<<<<<<< ###
def append_single(combo, single):
    single = create_single_dict(single)
    print('Retriving first compound ...')
    first = ['TYPE', 'CELL', 'COMP1', 'CONC1']
    combo = combo.groupby(by=first, as_index=False, sort=False).apply(lambda x: get_single(x, first, '1', single))
    print('Retriving second compound ...')
    second = ['TYPE', 'CELL', 'COMP2', 'CONC2']
    combo = combo.groupby(by=second, as_index=False, sort=False).apply(lambda x: get_single(x, second, '2', single))
    return combo

def get_single(df, col, suffix, single):
    info = df[col].iloc[0,:].squeeze()
    ratio, ratio_adj = single.get(tuple(info), (np.nan, np.nan))
    df['RATIO'+suffix] = ratio
    df['RATIOADJ'+suffix] = ratio_adj
    return df

def create_single_dict(df):
    eff = {(row['TYPE'], row['CELL'], row['COMP1'], row['CONC1']): (row['RATIO'], row['RATIOADJ']) for idx, row in df.iterrows()}
    return eff

def get_expected(x1, x2):
    score = np.array([np.nan] * len(x1))
    idx = (x1 <= 0) & (x2 <= 0)
    score[idx] = np.minimum(x1, x2)[idx]
    score[~idx] = (np.minimum(x1, 1) * np.minimum(x2, 1))[~idx]
    return score


#################################    main   ###################################
combo_path  = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.score.csv')
single_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.single.score.csv')
out_path    = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.syn.csv')

syn = cal_syn_score(single_path, combo_path)
syn.to_csv(out_path, index=None)