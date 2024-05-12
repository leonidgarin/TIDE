## What's TIDE?

TIDE - Time Invariant Discretization Engine

This is a novel algorithm for binning task that takes into account stability of bins over different time periods.
It divides mixed-type feature to groups (bins) and then uses Weight of Evidence to target-encode it.

The algorithm features:
- Automatic event rate trend detection
- Monotonic event rate over continuous bins with positive or negative trend
- Stability of event rates relative position over different time periods
- Adjustable minimal event rate delta between continuous bins 
- Minimal size of continuous bins
- Every continuous bin has minimum n observations of every class (0/1)
- Chi-merge for categorical bins
- Merge missing values to "best"/"worst" continuous bin depending on missing rate
- Calculation of Information Value, Population Stability Index and other stats
- Stats table and plots for binning result evaluation
- ...and more!

## Installation
Download .tar.gz to your local machine from [Latest Release Page](https://github.com/leonidgarin/TIDE/releases/latest), then `pip install tide-<version>.tar.gz` using terminal.

## Quick start
- Basic usage
```Python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tide import TIDE

# Define data
X = pd.DataFrame(<your factors>)
y = pd.Series(<your binary target>)
per = pd.Series(<your periods>)

# Fit groups
mytide = TIDE()
mytide.fit(X,y,per)

# Weight of Evidence transformation
X_woe = mytide.transform(X)

# Fit model and check ROC-AUC
lr = LogisticRegression()
lr.fit(X_woe,y)
pred = lr.predict_proba(X_woe)[::,1]
print(roc_auc_score(y,pred))
```
- Show results of binning
```Python
# Show stats for all variables and fitted groups
mytide.stats

# Show stats for all periods, variables and fitted groups 
mytide.stats_per

# Plot: distribution of groups and event rates; event rate over time; PSI over time
mytide.plot([<list of variables>])

# Resulting bins
mytide.exog_bins

# Resulting WoEs
mytide.exog_woe
```
## Functions
- fit
- transform
- fit_transform
- plot

## Userful stats
- stats
- stats_per
- exog_bins
- exog_woe