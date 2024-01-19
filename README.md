## What's TIDE?

TIDE - Time Invariant Discretization Engine

This is a novel algorithm for binning task that takes into account stability of bins over different time periods.

## Quick start
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tide import TIDE

X = pd.DataFrame(<your factors>)
y = pd.Series(<your binary target>)
per = pd.Series(<your periods>)

mytide = TIDE()
mytide.fit(X,y,per)

X_woe = mytide.transform(X)

lr = LogisticRegression()
lr.fit(X_woe,y)
pred = lr.predict_proba(X_woe)[::,1]
print(roc_auc_score(y,pred))
'''

## Functions
- fit
- transform
- fit_transform
- plot

## Known issues
- Stats does not show brackets properly for continuous bins

## Plan
- Add chi-squared to the algorithm
- 
