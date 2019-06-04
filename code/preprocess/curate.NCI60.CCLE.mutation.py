####################################################################################
###                            curate.NCI60.mutation.py                          ###
####################################################################################
proj_dir = '/work/bioinformatics/s418336/projects/DLSyn'
import os
import re
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import pandas as pd
import numpy as np
import utility.utility as util

##############################    function   ################################
def read_mutation_nci60(data_path, cell_path, mut_path, minMut=1, filter='all_gene', impute=False):
    cell = np.unique(pd.read_csv(data_path, usecols=['CELL'], squeeze=True))
    cell = list(map(util.cleanCellName, cell))
    cell_id = pd.read_csv(cell_path, usecols=['DepMap_ID', 'CCLE_Name'])
    cell_id['CCLE_Name'] = cell_id['CCLE_Name'].apply(lambda x: x.split('_')[0].replace('NIH', ''))
    cell_id = cell_id.loc[cell_id['CCLE_Name'].isin(cell),:]
    cell_id = {row['DepMap_ID']:row['CCLE_Name'] for _, row in cell_id.iterrows()}
    # read mutation
    mut = pd.read_csv(mut_path, usecols=['Hugo_Symbol', 'Variant_Classification', 'DepMap_ID'])
    mut = mut.loc[mut['DepMap_ID'].isin(cell_id.keys()),:]
    mut = mut.loc[mut['Variant_Classification'].notnull(),:]
    mut = mut.loc[~mut['Variant_Classification'].isin(['Silent', 'Intron', "3'UTR", "5'UTR", "5'Flank"]),:]
    mut = mut[['DepMap_ID', 'Hugo_Symbol']].drop_duplicates().rename(columns={'DepMap_ID': 'CELL', 'Hugo_Symbol': 'GENE'})
    if minMut > 1:
        genes = mut['GENE'].value_counts()
        genes = list(genes[genes > minMut].index)
        mut = mut.loc[mut['GENE'].isin(genes),:]
    if filter =='cancer_gene':
        genes = util.read_cancer_gene()
        mut = mut.loc[mut['GENE'].isin(genes),:]
    mut['VALUE'] = 1
    mut = mut.pivot(index='CELL', columns='GENE', values='VALUE').fillna(0)
    mut.index = mut.index.to_series().apply(lambda x: cell_id[x])
    # impute cell line
    if impute:
        no_cell = [c for c in cell if c not in mut.index]
        if len(no_cell) > 0:
            mut = pd.concat([mut, pd.DataFrame(0, index=no_cell, columns=mut.columns)])
    mut = mut.sort_index(axis=0).sort_index(axis=1)
    # outpath
    out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/CCLE_mutation.min_{}.{}.csv'.format(minMut, filter))
    mut.to_csv(out_path)

def read_mutation_all(data_path, cell_path, mut_path, nci60_mut):
    mut = pd.read_csv(mut_path, usecols=['Hugo_Symbol', 'Variant_Classification', 'DepMap_ID'])
    mut = mut.loc[mut['Variant_Classification'].notnull(),:]
    mut = mut.loc[~mut['Variant_Classification'].isin(['Silent', 'Intron', "3'UTR", "5'UTR", "5'Flank"]),:]
    mut = mut[['DepMap_ID', 'Hugo_Symbol']].drop_duplicates().rename(columns={'DepMap_ID': 'CELL', 'Hugo_Symbol': 'GENE'})
    genes = list(pd.read_csv(nci60_mut, index_col=0).columns)
    mut = mut.loc[mut['GENE'].isin(genes),:]
    mut['VALUE'] = 1
    # ccle cell
    cell_id = pd.read_csv(cell_path, usecols=['DepMap_ID', 'CCLE_Name'])
    cell_id['CCLE_Name'] = cell_id['CCLE_Name'].apply(lambda x: x.split('_')[0].replace('NIH', ''))
    cell_id = {row['DepMap_ID']:row['CCLE_Name'] for _, row in cell_id.iterrows()}
    # mut table
    mut = mut.pivot(index='CELL', columns='GENE', values='VALUE').fillna(0)
    mut.index = mut.index.to_series().apply(lambda x: cell_id[x])
    mut = mut.loc[~mut.index.to_series().str.contains('MERGE'),:]
    mut = mut.sort_index(axis=0)
    mut = mut[genes]
    # out path
    out_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/CCLE_mutation.cancer_gene.csv')
    mut.to_csv(out_path)

#############################      main     ###############################
data_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/ComboScore/curated.combo.syn.doseagg_min.csv')
cell_path = os.path.join(proj_dir, 'data/DepMap/DepMap.celllines.2019.q1.csv')
mut_path = os.path.join(proj_dir, 'data/DepMap/CCLE_Mutation_19Q1.csv')
# read_mutation_nci60(data_path, cell_path, mut_path, minMut=1, filter='cancer_gene', impute=False)
nci60_path = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/NCI60/CCLE_mutation.min_1.cancer_gene.csv')
read_mutation_all(data_path, cell_path, mut_path, nci60_path)