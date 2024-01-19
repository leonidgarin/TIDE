'''
Formulas and statistics calculation
- Weight of Evidence
- Information Value
- Event rates per bin
- other stuff

TODO:
- Update docs
'''

import numpy as np


def calc_eventrates(x,
                    y,
                    composed_bins,
                    idx_cont=None,
                    round_brackets='right',
                    epsilon=1e-6,
                    return_counts=False):
    '''
    Calculates rate of occurence of positive observations in each composed bin

    Arguments
    _________
    x : np.array
        Array of exogenous variable observations, can be mixed-type

    y : np.array
        Array of endogenous variable observations, must be binary

    composed_bins : dict
        A dict that contains composed bins. See example below:
        {0: {(-np.inf, 0.34): 'cont', 'special_category': 'cat'},
         1: {(0.34, 0.65): 'cont'},
         2: {(0.65, np.inf): 'cont'}}
        Key indicates incremental index of composed bin.
        The composed bin is a dict with atomic bin as key and its type as value.

    idx_cont : list-like or None
        Index of x that corresponds to its continuous values.
        If None, then all values are treated as continuous

    round_brackets : {'left','right'}, default='right'
        Used to determine the brackets treatment behavior.
        Note that it is expected that the minimum (and maximum)
        continuous bound across all composed bins has been adjusted
        by small delta.
        For example, if min bound is 1, the adjusted should be 1 - 1e-6;
        if max bound is 10, the adjusted should be 10 + 1e-6.
        
    return_counts : bool, default=False
        Indicates to return the numerators and denominators
        of event rates
    
    epsilon : float > 0, default=1e-6
        Small value that is used for safe math

    Returns
    _______
    eventrates [, events, totals] : np.array
    '''
    assert len(x) > 0
    assert len(x) == len(y)

    if isinstance(idx_cont,type(None)):
        idx_cont = [True] * len(x)

    totals = []
    events = []

    for idx, atomic_bins in composed_bins.items():

        n_total = 0
        n_events = 0

        for abin, bintype in atomic_bins.items():

            if bintype == 'cont':

                if round_brackets == 'right':
                    idx_bin = (x[idx_cont] >= abin[0]) & (x[idx_cont] < abin[1])

                elif round_brackets == 'left':
                    idx_bin = (x[idx_cont] > abin[0]) & (x[idx_cont] <= abin[1])

                else:
                    raise ValueError(f'Round brackets {round_brackets} not supported. Use "left" or "right"')

                n_total += y[idx_cont][idx_bin].shape[0]
                n_events += y[idx_cont][idx_bin].sum()

            elif bintype in ('cat','missing'):

                idx_bin = x == abin

                n_total += y[idx_bin].shape[0]
                n_events += y[idx_bin].sum()

            else:
                raise ValueError(f'Bin type {bintype} not supported')
            
        totals.append(n_total)
        events.append(n_events)

    totals = np.array(totals)
    events = np.array(events)
    eventrates = events / (totals + epsilon)

    if return_counts:
        return eventrates, events, totals
    else:
        return eventrates
    


def calc_WoEs(x,
              y,
              composed_bins,
              idx_cont=None,
              round_brackets='right',
              epsilon=1e-6):
    '''
    Calculates Weight of Evidence for every composed bin
    WoE = ln(%nonevents/%events)
    %events = events_bin / events_all
    %nonevents = nonevents_bin / nonevents_all

    Arguments
    _________
    x : np.array
        Array of exogenous variable observations, can be mixed-type

    y : np.array
        Array of endogenous variable observations, must be binary

    composed_bins : dict
        A dict that contains composed bins. See example below:
        {0: {(-np.inf, 0.34): 'cont', 'special_category': 'cat'},
         1: {(0.34, 0.65): 'cont'},
         2: {(0.65, np.inf): 'cont'}}
        Key indicates incremental index of composed bin.
        The composed bin is a dict with atomic bin as key and its type as value.

    idx_cont : list-like or None
        Index of x that corresponds to its continuous values.
        If None, then all values are treated as continuous

    round_brackets : {'left','right'}, default='right'
        Used to determine the brackets treatment behavior.
        Note that it is expected that the minimum (and maximum)
        continuous bound across all composed bins has been adjusted
        by small delta.
        For example, if min bound is 1, the adjusted should be 1 - 1e-6;
        if max bound is 10, the adjusted should be 10 + 1e-6.
    
    epsilon : float > 0, default=1e-6
        Small value that is used for safe math

    Returns
    _______
    WoEs : np.array
    '''
    assert len(x) > 0
    assert len(x) == len(y)
    assert y.sum() >= 1
    assert (1-y).sum() >= 1

    if isinstance(idx_cont,type(None)):
        idx_cont = [True] * len(x)

    totals = []
    events = []

    for idx, atomic_bins in composed_bins.items():

        n_total = 0
        n_events = 0

        for abin, bintype in atomic_bins.items():

            if bintype == 'cont':

                if round_brackets == 'right':
                    idx_bin = (x[idx_cont] >= abin[0]) & (x[idx_cont] < abin[1])

                elif round_brackets == 'left':
                    idx_bin = (x[idx_cont] > abin[0]) & (x[idx_cont] <= abin[1])

                else:
                    raise ValueError(f'Round brackets {round_brackets} not supported. Use "left" or "right"')

                n_total += y[idx_cont][idx_bin].shape[0]
                n_events += y[idx_cont][idx_bin].sum()

            elif bintype in ('cat','missing'):

                idx_bin = x == abin

                n_total += y[idx_bin].shape[0]
                n_events += y[idx_bin].sum()

            else:
                raise ValueError(f'Bin type {bintype} not supported')
            
        totals.append(n_total)
        events.append(n_events)

    totals = np.array(totals)
    events = np.array(events)
    nonevents = totals - events
    events_r = events / events.sum()
    nonevents_r = nonevents / nonevents.sum()

    WoEs = np.log((nonevents_r) / (events_r + epsilon))

    return WoEs



def calc_IVs(x,
             y,
             composed_bins,
             idx_cont=None,
             round_brackets='right',
             epsilon=1e-6):
    '''
    Calculates Information Value
    IV = sum((%nonevents - %events) * ln(%nonevents / %events))
    %events = events_bin / events_all
    %nonevents = nonevents_bin / nonevents_all

    Arguments
    _________
    x : np.array
        Array of exogenous variable observations, can be mixed-type

    y : np.array
        Array of endogenous variable observations, must be binary

    composed_bins : dict
        A dict that contains composed bins. See example below:
        {0: {(-np.inf, 0.34): 'cont', 'special_category': 'cat'},
         1: {(0.34, 0.65): 'cont'},
         2: {(0.65, np.inf): 'cont'}}
        Key indicates incremental index of composed bin.
        The composed bin is a dict with atomic bin as key and its type as value.

    idx_cont : list-like or None
        Index of x that corresponds to its continuous values.
        If None, then all values are treated as continuous

    round_brackets : {'left','right'}, default='right'
        Used to determine the brackets treatment behavior.
        Note that it is expected that the minimum (and maximum)
        continuous bound across all composed bins has been adjusted
        by small delta.
        For example, if min bound is 1, the adjusted should be 1 - 1e-6;
        if max bound is 10, the adjusted should be 10 + 1e-6.
    
    epsilon : float > 0, default=1e-6
        Small value that is used for safe math

    Returns
    _______
    IVs : np.array
        Calculated parts of IV formula for each composed bin.
        Sum of IVs results in total IV for factor.
    '''
    assert len(x) > 0
    assert len(x) == len(y)
    assert y.sum() >= 1
    assert (1-y).sum() >= 1

    if isinstance(idx_cont,type(None)):
        idx_cont = [True] * len(x)

    totals = []
    events = []

    for idx, atomic_bins in composed_bins.items():

        n_total = 0
        n_events = 0

        for abin, bintype in atomic_bins.items():

            if bintype == 'cont':

                if round_brackets == 'right':
                    idx_bin = (x[idx_cont] >= abin[0]) & (x[idx_cont] < abin[1])

                elif round_brackets == 'left':
                    idx_bin = (x[idx_cont] > abin[0]) & (x[idx_cont] <= abin[1])

                else:
                    raise ValueError(f'Round brackets {round_brackets} not supported. Use "left" or "right"')

                n_total += y[idx_cont][idx_bin].shape[0]
                n_events += y[idx_cont][idx_bin].sum()

            elif bintype in ('cat','missing'):

                idx_bin = x == abin

                n_total += y[idx_bin].shape[0]
                n_events += y[idx_bin].sum()

            else:
                raise ValueError(f'Bin type {bintype} not supported')
            
        totals.append(n_total)
        events.append(n_events)

    totals = np.array(totals)
    events = np.array(events)
    nonevents = totals - events
    events_r = events / events.sum()
    nonevents_r = nonevents / nonevents.sum()

    IVs = (nonevents_r - events_r) * np.log((nonevents_r) / (events_r + epsilon))

    return IVs



def calc_PSIs(x,
              y,
              per,
              composed_bins,
              idx_cont=None,
              round_brackets='right',
              epsilon=1e-6):
    ''' 
    Calculates Population Stability Index:
    PSI = sum((%bin_t0 - %bin_t1) * ln(%bin_t0 - %bin_t1))
    %bin_t0 = n_obs_bin_t0 / n_obs_t0
    %bin_t1 = n_obs_bin_t1 / n_obs_t1

    Arguments
    _________
    x : np.array
        Array of exogenous variable observations, can be mixed-type

    y : np.array
        Array of endogenous variable observations, must be binary
    
    per : np.array
        Array of values that mark the period of observation.
        Note that period names ordering must correspond
        to the alphanumeric order.

    composed_bins : dict
        A dict that contains composed bins. See example below:
        {0: {(-np.inf, 0.34): 'cont', 'special_category': 'cat'},
         1: {(0.34, 0.65): 'cont'},
         2: {(0.65, np.inf): 'cont'}}
        Key indicates incremental index of composed bin.
        The composed bin is a dict with atomic bin as key and its type as value.

    idx_cont : list-like or None
        Index of x that corresponds to its continuous values.
        If None, then all values are treated as continuous

    round_brackets : {'left','right'}, default='right'
        Used to determine the brackets treatment behavior.
        Note that it is expected that the minimum (and maximum)
        continuous bound across all composed bins has been adjusted
        by small delta.
        For example, if min bound is 1, the adjusted should be 1 - 1e-6;
        if max bound is 10, the adjusted should be 10 + 1e-6.
    
    epsilon : float > 0, default=1e-6
        Small value that is used for safe math

    Returns
    _______
    PSIs : np.array
    '''
    assert len(x) == len(y) == len(per)

    per_unique = np.unique(per)

    PSIs = [np.nan, ]

    for per_old, per_new in zip(per_unique[:-1],per_unique[1:]):
        _, _, n_totals_old = calc_eventrates(x = x[per==per_old],
                                             y = y[per==per_old],
                                             composed_bins = composed_bins,
                                             idx_cont = idx_cont[per==per_old],
                                             round_brackets = round_brackets,
                                             epsilon = epsilon,
                                             return_counts=True)
        
        rates_old = n_totals_old / (n_totals_old.sum() + epsilon)

        _, _, n_totals_new = calc_eventrates(x = x[per==per_new],
                                             y = y[per==per_new],
                                             composed_bins = composed_bins,
                                             idx_cont = idx_cont[per==per_new],
                                             round_brackets = round_brackets,
                                             epsilon = epsilon,
                                             return_counts=True)
        
        rates_new = n_totals_new / (n_totals_new.sum() + epsilon)

        PSI_i =  np.sum((rates_new-rates_old) * np.log(rates_new/(rates_old + epsilon)))
        PSIs.append(PSI_i)

    return PSIs