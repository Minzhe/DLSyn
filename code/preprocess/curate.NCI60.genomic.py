####################################################################################
###                              curate.NCI60.expr.py                            ###
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
def read_genomic_data(data_path, out_dir):
    data = pd.read_csv(data_path, index_col=0).T
    data.index = data.index.to_series().apply(lambda x: util.cleanCellName(x.split(':')[-1]))
    data = data.sort_index(axis=0).sort_index(axis=1)
    data = data.loc[:,data.columns.to_series() != '-']
    out_path = os.path.join(out_dir, os.path.basename(data_path).strip('csv') + 'all_gene.csv')
    data.to_csv(out_path)
    # filter cancer gene
    genes = util.read_cancer_gene()
    data = data.loc[:,data.columns.to_series().isin(genes)]
    out_path = os.path.join(out_dir, os.path.basename(data_path).strip('csv') + 'cancer_gene.csv')
    data.to_csv(out_path)

def read_genomic_data_shuffle(data_path, out_dir):
    data = pd.read_csv(data_path, index_col=0).T
    data.index = data.index.to_series().apply(lambda x: util.cleanCellName(x.split(':')[-1]))
    data = data.sort_index(axis=0).sort_index(axis=1)
    data = data.loc[:,data.columns.to_series() != '-']
    # shuffle data
    for col in data.columns:
        # print(np.random.permutation(data[col]))
        data.loc[:,col] = np.random.permutation(data[col])
    # output
    out_path = os.path.join(out_dir, os.path.basename(data_path).strip('csv') + 'all_gene.shuffle.csv')
    data.to_csv(out_path)
    # filter cancer gene
    genes = util.read_cancer_gene()
    data = data.loc[:,data.columns.to_series().isin(genes)]
    out_path = os.path.join(out_dir, os.path.basename(data_path).strip('csv') + 'cancer_gene.shuffle.csv')
    data.to_csv(out_path)

#############################      main     ###############################
expr_path = os.path.join(proj_dir, 'data/NCI.CellMiner/RNA_seq_composite_expression.csv')
mut_path = os.path.join(proj_dir, 'data/NCI.CellMiner/DNA_Exome_Seq_protein_function_affecting.csv')
protein_path = os.path.join(proj_dir, 'data/NCI.CellMiner/SWATH_Protein.csv')
out_dir = os.path.join(proj_dir, 'data/Curated/ALMANAC/Genomics/')

# read_genomic_data(expr_path, out_dir)
# read_genomic_data(mut_path, out_dir)
# read_genomic_data(protein_path, out_dir)
# read_genomic_data_shuffle(expr_path, out_dir)
# read_genomic_data_shuffle(mut_path, out_dir)
# read_genomic_data_shuffle(protein_path, out_dir)