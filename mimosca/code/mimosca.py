from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
import seaborn as sns
import time

y = pd.read_csv("../data/day0/exp_singlets.tsv", sep="\t")
cells_y = y['cell']
y = y.drop(columns=['cell'])

X = pd.read_csv("../data/day0/design_guide_counts.tsv", sep="\t")
cells_x = X['cell']
X = X.drop(columns=['cell'])

mimosca = ElasticNet(l1_ratio = 0.5,alpha = 0.0005,max_iter = 10000)
coefs = mimosca.fit(X, y[list(y.columns)[0]]).sparse_coef_.todense()
for g in list(y.columns)[1:]:
    coefs = np.vstack((coefs, mimosca.fit(X, y[g]).sparse_coef_.todense()))
coef_table = pd.DataFrame(coefs)
coef_table.columns = X.columns
coef_table['gene'] = y.columns

cols = coef_table.columns.tolist()
cols = cols[-1:] + cols[:-1]
coef_table = coef_table[cols]

coef_table.to_csv("../data/day0/mimosca_coefs.tsv", sep="\t", index=False)
