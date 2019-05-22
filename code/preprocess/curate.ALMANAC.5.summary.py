#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import quantile_transform

#########################      function     ##########################
def summarize_matrix(path, agg='mean', normalize=False):
    data = pd.read_csv(path)
    out_path = path.strip('.csv') + '.{}.norm_{}.csv'.format(agg, str(normalize))
    if agg == 'mean':
        data = data.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False)[['RATIO_SYN', 'RATIOADJ_SYN']].mean()  
    elif agg == 'max':
        data = data.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False)[['RATIO_SYN', 'RATIOADJ_SYN']].max()
    elif agg == 'min':
        data = data.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False)[['RATIO_SYN', 'RATIOADJ_SYN']].min()
    elif agg == 'min_max_mean':
        data = data.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False)[['RATIO_SYN', 'RATIOADJ_SYN']].agg(['min', 'max', 'mean'])
        data.columns = [score + '_' + val for score, val in zip(list(data.columns.get_level_values(0)), list(data.columns.get_level_values(1)))]
        data = data.reset_index()
    elif agg == 'dose':
        data['COMP1'] = data['COMP1'].astype(int).astype(str)
        data['COMP2'] = data['COMP2'].astype(int).astype(str)
        data['CONC1'] = data['CONC1'].astype(str)
        data['CONC2'] = data['CONC2'].astype(str)
        data['COMP1'] = data[['COMP1', 'CONC1']].apply(lambda x: '@'.join(x), axis=1)
        data['COMP2'] = data[['COMP2', 'CONC2']].apply(lambda x: '@'.join(x), axis=1)
        data = data[['TYPE', 'CELL', 'COMP1', 'COMP2', 'RATIO_SYN', 'RATIOADJ_SYN']]
    else:
        raise ValueError('Unrecognizable agg function.')
    data = data.dropna()
    if normalize:
        data[['RATIO_SYN', 'RATIOADJ_SYN']] = quantile_transform(data[['RATIO_SYN', 'RATIOADJ_SYN']], axis=0, output_distribution='normal', n_quantiles=data.shape[0], subsample=data.shape[0])
    data.to_csv(out_path, index=False)


#################################    main   ###################################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.syn.csv')
# summarize_matrix(data_path, agg='mean', normalize=True)
# summarize_matrix(data_path, agg='max')
summarize_matrix(data_path, agg='min')
# summarize_matrix(data_path, agg='min_max_mean')
# summarize_matrix(data_path, agg='dose')