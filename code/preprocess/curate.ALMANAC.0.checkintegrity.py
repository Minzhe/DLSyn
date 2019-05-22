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

def check_score(path):
    data = pd.read_csv(path, usecols=['NSC1', 'CONCINDEX1', 'CONC1', 'NSC2', 'CONCINDEX2', 'CONC2', 'TESTVALUE', 'CONTROLVALUE', 'TZVALUE', 'PERCENTGROWTH', 'EXPECTEDGROWTH', 'SCORE', 'PANEL', 'CELLNAME'], nrows=100000)
    # data = pd.read_csv(path, usecols=)
    data['RATIOADJ'] = (data['TESTVALUE'] - data['TZVALUE']) / (data['CONTROLVALUE'] - data['TZVALUE'])
    data['PERCENTGROWTH'] = data['PERCENTGROWTH']/100
    data = data[['TESTVALUE', 'CONTROLVALUE', 'TZVALUE', 'PERCENTGROWTH', 'RATIOADJ']]
    return data

### >>>>>>>>>>>>>>>>>>>>>> utility <<<<<<<<<<<<<<<<<<<<<< ###
def count_row(df):
    n = df.shape[0]
    df['NUMEXPR'] = pd.Series(n * [n], index=df.index)
    return df


##############################    main   ################################
combo_path = os.path.join(proj_dir, 'data/NCI.ALMANAC/ComboDrugGrowth_Nov2017.csv')
out_path   = os.path.join(proj_dir, 'data/NCI.ALMANAC/test.csv')
combo = check_score(combo_path)
combo.to_csv(out_path, index=None)