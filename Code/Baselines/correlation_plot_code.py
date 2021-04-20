# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 01:02:33 2021

@author: Sarwan
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_tmp = np.load("rodge_correlation_matrix.npy")
asd = pd.DataFrame(data_tmp)
sns.heatmap(asd.corr());

plt.savefig('ridge.png', bbox_inches='tight', pad_inches=0.0)