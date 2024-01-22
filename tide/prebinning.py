'''
Algorithms for pre-binning
'''
import numpy as np


def equal_size_binning_cont(x,
                       n_bins,
                       min_bound = -np.inf,
                       max_bound = np.inf,
                       epsilon = 1e-8):
    '''
    Quantile binning for continuous variables

    Arguments
    _________
    x : list-like
        List of observations of continuous variable

    n_bins : int >= 2
        Number of bins to create

    min_bound : float or 'col_min'
        A minimum bound for first continuous bin.
        If 'col_min', than the minimum column value is used.

    max_bound : float or 'col_max' (default = np.inf)
        A maximum bound for last continuous bin.
        If 'col_max', than the maximum column value is used.

    epsilon : float, default = 1e-8
        Small number to adjust first and last bound

    Returns
    _______
    bins_map
        A dict with continuous bins and their indexes.
        Example:
        {'bins':
            {'trend':'unknown',
             'cont':
                {(-np.inf,0.45):0,
                 (0.45,3.29):1,
                 (3.29,10.33):2,
                 (10.33,np.inf):3
                }
            }
        }
    '''
    assert len(x) > 0
    assert n_bins >= 2
    assert min_bound in ('col_min',) or min_bound <= min(x)
    assert max_bound in ('col_max',) or max_bound >= max(x)

    q = np.linspace(0,100,n_bins+1)
    bounds = np.percentile(x,q)

    if min_bound != 'col_min':
        bounds[0] = min_bound - epsilon
    if max_bound != 'col_max':
        bounds[-1] = max_bound + epsilon

    bins_map = {'trend':'unknown','bins':{'cont':{b:idx for idx,b in enumerate(zip(bounds[:-1],bounds[1:]))}}}
    
    return bins_map