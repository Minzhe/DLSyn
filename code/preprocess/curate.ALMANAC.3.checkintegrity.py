#############################################################################
###                            curate.ALMANAC.py                          ###
#############################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import pandas as pd
import numpy as np

##############################    function   ################################
def check_integrity(path):
    combo = pd.read_csv(path)
    combo = combo.loc[combo['SCORE'].notnull(),:]
    # check matrix integrity
    combo = combo.groupby(by=['CELL', 'COMP1', 'COMP2']).apply(count_row)
    combo = combo.loc[combo['NUMEXPR'] != 9,:]
    return combo


### >>>>>>>>>>>>>>>>>>>>>> utility <<<<<<<<<<<<<<<<<<<<<< ###
def count_row(df):
    n = df.shape[0]
    df['NUMEXPR'] = pd.Series(n * [n], index=df.index)
    return df


##############################    main   ################################
combo_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/raw.combo.growth.csv')
out_path   = os.path.join(proj_dir, 'data/Curated/ALMANAC/test.csv')
combo = check_integrity(combo_path)
combo.to_csv(out_path, index=None)