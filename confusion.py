import sys
import numpy as np
import scipy

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns

y_act_raw = open(sys.argv[1],'rt')
y_act = np.loadtxt(y_act_raw, dtype = 'str', delimiter="\n")

y_act = np.array(y_act)

y_raw = open(sys.argv[2],'rt')
y = np.loadtxt(y_raw, dtype = 'str', delimiter="\n")

y = np.array(y)

confusion = confusion_matrix(y_act, y, labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'])

print(confusion)

f1 = f1_score(y_act, y, labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'], average = None)

print(f1)

micro = f1_score(y_act, y, labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'], average = 'micro')

print(micro)

macro = f1_score(y_act, y, labels=['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior'], average = 'macro')

print(macro)

ax = sns.heatmap(confusion, linewidth=0.5)
plt.show()
