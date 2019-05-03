#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np

#########################      function     ##########################
def summarize_matrix(path, agg='mean'):
    data = pd.read_csv(path)
    out_path = path.strip('.csv') + '.{}.csv'.format(agg)
    if agg == 'mean':
        data = data.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False).mean()  
    elif agg == 'max':
        data = data.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False).max()
    elif agg == 'min':
        data = data.groupby(by=['TYPE', 'CELL', 'COMP1', 'COMP2'], as_index=False, sort=False).min()
    else:
        raise ValueError('Unrecognizable agg function.')
    data = data.drop(['CONC1', 'CONC2'], axis=1)
    data.to_csv(out_path, index=False)


#################################    main   ###################################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/curated.combo.syn.csv')
summarize_matrix(data_path, agg='mean')
summarize_matrix(data_path, agg='max')
summarize_matrix(data_path, agg='min')