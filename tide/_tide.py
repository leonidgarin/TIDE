'''
TIDE - Time Invariant Discretization Engine

Leonid Garin, 2023

TODO:
- fix visual representation of bin brackets
- add multiprocessing to bins calculation
- write more tests
- mandatory missing as worst/best when no missing values in column (add parameter)
- cover case when cont bin contains only one unique value
'''

import warnings

import numpy as np 
import scipy.stats as ss 
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from .prebinning import equal_size_binning_cont
from .calc import (calc_eventrates, calc_WoEs, calc_IVs, calc_PSIs)



class TIDE:

    def __init__(self,
                 min_sample_rate = 0.05,
                 min_class_obs = 1,
                 min_bound = -np.inf,
                 max_bound = np.inf,
                 min_er_delta = 0.0,
                 n_prebins = 'log',
                 missing_categories = [],
                 missing_strategy = 'separate_bin',
                 missing_rate = 0.05,
                 cat_strategy = 'separate_bin',
                 alpha_significance = 0.05
                 ):
        '''
        TIDE - Time Invariant Discretization Engine

        Binning is a technique to group values of exogenous variable.
        After grouping, bins are used to target-encode the variable
        with Weight of Evidence (WoE) values. Then WoE-transformed
        variable can be used as feature for logistic regression.
        This allows to build robust and interpretable models, which is
        useful, for example, to create credit scorecards.

        The TIDE binning is performed on arrays containing mixed-type
        points: integers, floats, strings. The numbers are
        treated as continuous part, while strings as categorical;
        some strings as missing values.

        TIDE provides and guarantees the following restrictions for bins:
        - The trend of event rate in each subsequent bin
        must be monotonical (positive or negative);
        - This trend must remain the same on different time
        intervals within the training data;
        - A bin must be at least 5%* size of total data
        observations;
        - A bin must contain at least 1* observation of
        both classes;
        - Missing values must be taken into account: placed in
        separate bin if there are sufficient cases, or joined
        to some other bin (the "worst" / the "best").
        * values can be adjusted.

        Attributes
        __________
        min_sample_rate : float (0.0 < float < 1.0, default = 0.05)
            Specified minumum number of bin observations
            divided by total number of observations in dataset.

        min_class_obs : int (int >= 1, default = 1)
            Minimum number of observations of each class
            for all bins.

        min_bound : float or 'col_min' (default = -np.inf)
            Lower bound for first continuous bin.
            If 'col_min', then the minimum column value is used.

        max_bound : float or 'col_max' (default = np.inf)
            A maximum bound for last continuous bin.
            If 'col_max', then the maximum column value is used.

        min_er_delta : float (default = 0.0)
            A minimum delta between event rates of adjacent continuous bins

        n_prebins : 'log' or int (int >= 2, default = 'log')
            Number of bins specified for prebinning algorithm,
            which is Quantile binning. If 'log', then the following
            formula is used:
            n_prebins = max(3,ceil(log(N_total)))

        missing_categories : list of str
            This list must contain categories that must be treated as missing.
            Note that np.nan is not allowed because it has unpredictable
            behavior (f.e. np.nan != np.nan).

        missing_strategy : {'separate_bin', 'worst_bin', 'best_bin'} (default = 'separate_bin')
            Defines the algorithm behavior on how to treat missing values.
            If 'separate_bin', then there will be special bin
            for missing values regardless of its size (must be at least 1 observation,
            else there will no missing bin).
            If 'worst_bin', missings will be included in the bin with
            highest event rate in case when missing_rate is lower than specified.
            If 'best_bin', missings will be included in the bin with
            lowest event rate in case when missing_rate is lower than specified.

        missing_rate : float (0.0 < float < 1.0) (default = 0.05)
            (Not implemented yet)
            If missing_rate is higher (>=) than specified,
            the algorithm will create a separate bin for missings.
            Else, the algorithm will create composed bin according to chosen strategy.
            For 'worst_bin' and ascending trend, missings will be included in last bin.
            For 'worst_bin' and descending trend, missing will be included in first bin.
            For 'best_bin' and ascending trend, missings will be included in first bin.
            For 'best_bin' and descending trend, missing will be included in last bin.

        cat_strategy : {'separate_bin','chi-squared','tide'} (default = 'separate_bin')
            This option specifies how the algorithm handles the
            categorical bins. Note that these strategies will not
            ensure stability of categorical bins over time.
            If 'separate_bin', then they will be kept as is.
            If 'chi-squared', then the algorithm will loop merge categorical bins
            using chi-squared test (see ChiMerge by R.Kerber, 1992).
            If 'tide', then algorithm will loop merge categorical and continuous bins
            using square distance between event rate curves until no curve intersections
            present. Note: If one wants to include missings to achieve time stability

        alpha_significance : float, (0.0 < float < 0.5, default = 0.05)
            The specified threshold for p-value to reject
            the null hypothesis. Used for all chi-squared and other
            tests if applicable.

        Methods
        _______
        idx_from_mixed(x):
            Takes mixed-type column and return boolean indexes
            for its continuous, categorical and missing values

        fit(X, y, per, reset=True, return_bins=False, disable_tqdm=False):
            Fit one or many variables

        transform(self,X, xnames=None):
            Transform variable using previously calculated bins

        fit_transform(self,X,y,per,reset=True,return_bins=False):
            Scikit-learn style fit and transform of one or many variables
        
        plot(self,xnames = [],figsize=(15,6),dpi=120):
            Matplotlib plot for variable binning results
        '''
        # Constants
        self._epsilon = 1e-6

        self._acceptable_types_cont = (int,np.int8,np.int16,np.int32,np.int64,
                                        float,np.float16,np.float32,np.float64)
        
        self._acceptable_types_cat = (str,)

        self._acceptable_types = self._acceptable_types_cont + self._acceptable_types_cat

        # Storages
        self.exog_bins = dict()
        self.exog_woes = dict()
        self.stats = pd.DataFrame(columns = ['variable',
                                             'trend',
                                             'bin',
                                             'n_obs',
                                             'n_events',
                                             'event_rate',
                                             'WoE',
                                             'IV_contrib',
                                             'IV_total'])
        self.stats_per = pd.DataFrame(columns = ['period',
                                                 'variable',
                                                 'trend',
                                                 'bin',
                                                 'n_obs',
                                                 'n_events',
                                                 'event_rate',
                                                 'PSI_seq',
                                                 'PSI_base'])
        
        # Behavior arguments       
        assert 0.0 < min_sample_rate < 1.0
        self.min_sample_rate = min_sample_rate

        assert min_class_obs >= 1
        self.min_class_obs = min_class_obs

        if min_bound != 'col_min':
            assert isinstance(min_bound,self._acceptable_types_cont) and pd.isna(min_bound) == False
        self.min_bound = min_bound

        if max_bound != 'col_max':
            assert isinstance(max_bound,self._acceptable_types_cont) and pd.isna(max_bound) == False
        self.max_bound = max_bound

        assert 0.0 <= min_er_delta <= 1.0
        self.min_er_delta = min_er_delta

        if n_prebins != 'log':
            assert n_prebins >= 2
        self.n_prebins = n_prebins

        assert (isinstance(missing_categories,(list,tuple,set,np.ndarray))
                and all([isinstance(i,self._acceptable_types_cat) for i in missing_categories]))
        self.missing_categories = missing_categories

        assert missing_strategy in {'separate_bin','worst_bin','best_bin'}
        self.missing_strategy = missing_strategy

        assert 0.0 < missing_rate < 1.0
        self.missing_rate = missing_rate

        assert cat_strategy in {'separate_bin','chi-squared','tide'}
        self.cat_strategy = cat_strategy

        assert 0.0 < alpha_significance < 0.5
        self.alpha_significance = alpha_significance


    def _validate_naninf(self,x):
        if np.any(pd.isna(x)) or x[x==np.inf].shape[0]>0:
            raise ValueError('Input contains NaN and/or inf')


    def idx_from_mixed(self,x):
        '''
        Takes mixed-type column and returns boolean indexes
        for its continuous, categorical and missing values
        '''
        idx_cont = np.array([isinstance(val, self._acceptable_types_cont)\
                             and not val in self.missing_categories\
                             for val in x])
        idx_cat = np.array([isinstance(val,self._acceptable_types_cat)\
                            and not val in self.missing_categories\
                            for val in x])
        idx_missing = np.array([val in self.missing_categories\
                                for val in x])
        
        return idx_cont, idx_cat, idx_missing
    

    def _compose_bins(self,bins_map):
        '''
        Takes bins_map and returns composed bins        
        '''
        composed = {'trend':bins_map['trend'],'bins':dict()}

        for bin_type in ('cont','cat','missing'):
            binmap_i = bins_map['bins'].get(bin_type,dict())
            for b, idx in binmap_i.items():
                composed['bins'][idx] = composed['bins'].get(idx,dict())
                composed['bins'][idx][b] = bin_type

        return composed


    def _fit_single(self,x,y,per,xname):
        '''
        Calculates bins for single exogenous column.
        Also writes them to storage and launches stats calculation

        Arguments
        _________
        x : list-like
            A mixed-type array of observations of exogenous variable
        y : list-like
            Array of observations of binary endogenous variable
        per : list-like
            Array with values that indicate observation period
        xname : str
            Name of exogenous variable
        '''
        # Transform input to numpy arrays
        x_arr = np.array(x,dtype='O')
        y_arr = np.array(y)
        per_arr = np.array(per)

        per_unique = np.unique(per_arr)
        
        # Split x into continuous, discrete and missing parts
        idx_cont, idx_cat, idx_missing = self.idx_from_mixed(x_arr)

        x_cont = x_arr[idx_cont]
        y_cont = y_arr[idx_cont]
        per_cont = per_arr[idx_cont]

        x_missing = x_arr[idx_missing]
        y_missing = y_arr[idx_missing]
        per_missing = per_arr[idx_missing]

        # Determine trend based on equal size binning of continuous part
        if np.any(idx_cont): 

            # Calculate size of pre-bins
            if self.n_prebins == 'log':
                n_prebins = max(3,int(np.ceil(np.log(len(x)))))
            else:
                n_prebins = self.n_prebins

            # Pre-binning
            prebins_map = equal_size_binning_cont(x_cont,
                                                  n_prebins,
                                                  min_bound='col_min',
                                                  max_bound='col_max')
            prebins = self._compose_bins(prebins_map)['bins']

            if len(prebins.keys()) <= n_prebins and len(prebins.keys()) >= 2:
                n_prebins = len(prebins.keys())

                # Trend as sign of a1 in pairwise linear regression ER = a0 + a1 * bin_index
                # a1 > 0 = positive trend, else negative.
                prebins_eventrates = calc_eventrates(x_cont,y_cont,prebins)
                a1, _ = np.polyfit(np.arange(n_prebins),prebins_eventrates,deg=1)

                if np.isclose(a1,0):
                    a1 = 1
            else:
                a1 = 1  # There is one continuous bin.
                        # Almost certainly, it means that the continuous part
                        # contains one (or too little) distinct values.
        else:  # No continuous part
            a1 = 1

        if a1 > 0:
            trend = 'pos'
        else:
            trend = 'neg'


        # Select bound points for continuous bins
        cont_bounds = []
        included_missing = False
        if np.any(idx_cont):

            # Calculate crosstab for sorted unique x (continuous part) and periods
            # Every period has three columns: count of observations,
            # count of events per unique x, count of non-events

            x_cont_unique, idx_x_cont_unique = np.unique(x_cont,
                                                         return_inverse=True)  # According to the NumPy docs
                                                                               # (see https://numpy.org/doc/stable/reference/generated/numpy.unique.html),
                                                                               # the np.unique function also sorts x in ascending order
            x_unique = x_cont_unique #?
            idx_x_unique = idx_x_cont_unique
            per_x = per_cont
            y_x = y_cont


            if self.missing_strategy in ('worst_bin','best_bin'): # adding missing values to worst/best bins if missing_rate < k%

                r_missing = x_missing.shape[0] / x_arr.shape[0]
    
                if 0 < r_missing < self.missing_rate:

                    included_missing = True
                    x_missing_unique, idx_x_missing_unique = np.unique(x_missing,return_inverse=True)
                    
                    if ((self.missing_strategy == 'worst_bin' and trend == 'pos')
                        or (self.missing_strategy == 'best_bin' and trend == 'neg')):

                        x_unique = np.concatenate((x_unique, x_missing_unique), dtype='O')
                        idx_x_unique = np.concatenate((idx_x_unique, idx_x_missing_unique + idx_x_unique.max() + 1))
                        per_x = np.concatenate((per_x, per_missing))
                        y_x = np.concatenate((y_x, y_missing))

                    elif ((self.missing_strategy == 'best_bin' and trend == 'pos')
                        or (self.missing_strategy == 'worst_bin' and trend == 'neg')):

                        x_unique = np.concatenate((x_missing_unique, x_unique),dtype='O')
                        idx_x_unique = np.concatenate((idx_x_missing_unique, idx_x_unique + idx_x_missing_unique.max() + 1))
                        per_x = np.concatenate((per_missing, per_x))
                        y_x = np.concatenate((y_missing, y_x))

            ct = x_unique

            per_size = []

            for p_i in per_unique:
                p_counts = np.bincount(idx_x_unique,weights = per_x==p_i)
                ct = np.vstack((ct,p_counts))
                p_events = np.bincount(idx_x_unique,weights = (per_x==p_i)*y_x)
                ct = np.vstack((ct,p_events))
                p_nonevents = p_counts - p_events
                ct = np.vstack((ct,p_nonevents))
                per_size.append(y_arr[per_arr==p_i].shape[0])

            ct = ct.T

            per_size = np.array(per_size)

            idx_bounds = []
            best_cand = None
            idx_base = 0
            idx_max = x_unique.shape[0] - 1

            if trend == 'pos':
                er_previous = np.zeros(shape=per_unique.shape)
            else:
                er_previous = np.ones(shape=per_unique.shape) + self.min_er_delta

            # Cycle for continuous values
            while idx_base <= idx_max:

                # Calculate cumulative event rate for every period
                er_cumul = ct[idx_base:, 2::3].cumsum(axis=0) / (self._epsilon
                                                                 + ct[idx_base:, 1::3].cumsum(axis=0))

                # Calculate right window function according to trend
                if trend == 'pos':
                    er_right_window = np.minimum.accumulate(er_cumul[::-1],axis=0)[::-1]
                else:
                    er_right_window = np.maximum.accumulate(er_cumul[::-1],axis=0)[::-1]

                # Choose candidate points that ensure all the conditions for continuous part are met:
                # Monotonicity for every period
                cand_monotone = np.all(er_cumul == er_right_window, axis=1)

                # Candidate bin has equal or more than min_sample_rate * 100% observations out of total
                cand_binsize = np.all((ct[idx_base:, 1::3].cumsum(axis=0) /
                                      (self._epsilon + per_size)) >= self.min_sample_rate, axis=1)

                # Every class is present with at least n=min_class_obs observations
                cand_classprecense = np.all(ct[idx_base:, 2::3].cumsum(axis=0) >= self.min_class_obs, axis=1) \
                                     & np.all(ct[idx_base:, 3::3].cumsum(axis=0) >= self.min_class_obs, axis=1)
                
                # New bin eventrate is higher (or lower, according to trend) by k% than previous one
                if trend == 'pos':
                    cand_erdelta = np.all(er_cumul >= (er_previous + self.min_er_delta), axis = 1)
                else:
                    cand_erdelta = np.all(er_cumul <= (er_previous - self.min_er_delta), axis = 1)

                # Can't use missing value as candidate
                if included_missing:
                    cand_nonmissing = ~np.isin(ct[idx_base:,0],x_missing_unique)
                else:
                    cand_nonmissing = np.tile(True,ct[idx_base:].shape[0])
                
                # Choose candidate point
                cand_allchecks = np.all((cand_monotone,
                                         cand_binsize,
                                         cand_classprecense,
                                         cand_erdelta,
                                         cand_nonmissing),
                                        axis=0)
                print('idx_base',idx_base)
                if cand_allchecks.max() == False:
                    print('No candidate points found')
                    break #No candidates found, quit cycle
                else:
                    best_cand = np.argmax(cand_allchecks)
                # Accept point if remaining space satisfies the conditions
                idx_base_rem = idx_base + best_cand + 1
                if idx_base_rem >= ct.shape[0]:
                    print('no shape remaining')
                    print(idx_base,x_unique[idx_base])
                    print(idx_base + best_cand,x_unique[idx_base + best_cand])
                    print(idx_base + best_cand + 1,x_unique[idx_base + best_cand + 1])
                    break
                
                flag_rem_binsize = np.any(np.all((ct[idx_base_rem:, 1::3].cumsum(axis=0) /
                                                 (self._epsilon + per_size))
                                                 >= self.min_sample_rate,
                                                 axis=1))
                
                flag_rem_classprecense = np.any(np.all(ct[idx_base_rem:, 2::3].cumsum(axis=0)
                                                       >= self.min_class_obs, axis=1) \
                                                & np.all(ct[idx_base_rem:, 3::3].cumsum(axis=0)
                                                       >= self.min_class_obs, axis=1))
                
                if trend == 'pos':
                    flag_rem_erdelta = np.all((ct[idx_base_rem:, 2::3].sum(axis=0) /
                                                         (self._epsilon
                                                          + ct[idx_base_rem:, 1::3].sum(axis=0)))
                                                        >= er_cumul[best_cand] + self.min_er_delta)
                else:
                    flag_rem_erdelta = np.all((ct[idx_base_rem:, 2::3].sum(axis=0) /
                                                         (self._epsilon
                                                          + ct[idx_base_rem:, 1::3].sum(axis=0)))
                                                        <= er_cumul[best_cand] - self.min_er_delta)
                print('Save point if all true',{'flag_rem_binsize':flag_rem_binsize, 'flag_rem_classprecense':flag_rem_classprecense, 'flag_rem_erdelta':flag_rem_erdelta})
                # Save point and update base index 
                if all((flag_rem_binsize, flag_rem_classprecense, flag_rem_erdelta)):
                    if trend == 'pos':
                        idx_bounds.append(idx_base+best_cand+1)
                    else:
                        idx_bounds.append(idx_base+best_cand)
                    print('point saved',idx_bounds,x_unique[idx_bounds])
                    er_previous = er_cumul[best_cand]
                    idx_base += best_cand + 1
                else: #Quit main cycle
                    break

            # Create bins
            if self.min_bound == 'col_min':
                min_bound = x_cont_unique.min() - self._epsilon
            else:
                min_bound = self.min_bound
            if self.max_bound == 'col_max':
                max_bound = x_cont_unique.max() + self._epsilon
            else:
                max_bound = self.max_bound

            mid_bounds = x_unique[idx_bounds]

            if len(mid_bounds)>0 and trend == 'pos' and mid_bounds[0] == x_unique[0]:
                mid_bounds = mid_bounds[1:]
            if len(mid_bounds)>0 and trend == 'neg' and mid_bounds[-1] == x_unique[-1]:
                mid_bounds = mid_bounds[:-1]

            cont_bounds = np.r_[min_bound, mid_bounds, max_bound]
            print('resulting bounds', cont_bounds)

        # Compose final bins
        bins_map = {'trend':trend, 'bins':dict()}
        idx = 0

        #continuous
        for val in zip(cont_bounds[:-1],cont_bounds[1:]):
            bins_map['bins']['cont'] = bins_map['bins'].get('cont',dict())
            bins_map['bins']['cont'][val] = idx
            idx += 1

        if included_missing: # add missings to best/worst bin
            if ((self.missing_strategy == 'worst_bin' and trend == 'pos')
                or (self.missing_strategy == 'best_bin' and trend == 'neg')):
                for val in x_missing_unique:
                    bins_map['bins']['missing'] = bins_map['bins'].get('missing',dict())
                    bins_map['bins']['missing'][val] = idx - 1

            elif ((self.missing_strategy == 'best_bin' and trend == 'pos')
                or (self.missing_strategy == 'worst_bin' and trend == 'neg')):
                for val in x_missing_unique:
                    bins_map['bins']['missing'] = bins_map['bins'].get('missing',dict())
                    bins_map['bins']['missing'][val] = 0

        #categorical
        bins_cat = np.unique(x[idx_cat])
        if self.cat_strategy == 'separate_bin':
            for val in bins_cat:
                bins_map['bins']['cat'] = bins_map['bins'].get('cat',dict())
                bins_map['bins']['cat'][val] = idx
                idx += 1

        elif self.cat_strategy == 'chi-squared':

            if len(bins_cat) >= 100:
                warnings.warn(f'Column {xname} has {len(bins_cat)} unique categorical values.\nChi-merge for categorical values has O(n^2) complexity, so performance may be poor')

            if len(bins_cat) > 0:
                bins_chi = [(i,) for i in bins_cat]
                pvalue_map = dict()

                # Initial p-value calculation
                for bin_i in bins_chi:

                    for bin_j in bins_chi:

                        if bin_i == bin_j or frozenset((bin_i,bin_j)) in pvalue_map.keys():
                            continue

                        n_i_0 = (y[np.isin(x,bin_i)]==0).sum()
                        n_i_1 = (y[np.isin(x,bin_i)]==1).sum()
                        n_j_0 = (y[np.isin(x,bin_j)]==0).sum()
                        n_j_1 = (y[np.isin(x,bin_j)]==1).sum()

                        if ((n_i_0, n_i_1, n_j_0, n_j_1).count(0) >= 2
                            and not ((n_i_0 == 0 and n_i_1 > 0 and n_j_0 > 0 and n_j_1 == 0)
                                     or (n_i_0 > 0 and n_i_1 == 0 and n_j_0 == 0 and n_j_1 > 0))):
                            pvalue = 1
                        else:
                            pvalue = ss.chi2_contingency([[n_i_0, n_i_1],
                                                          [n_j_0, n_j_1]]).pvalue
                            
                        pvalue_map[frozenset((bin_i,bin_j))] = pvalue

                # Chi-merge cycle
                while True:
                    
                    # Find maximum p-value between two bins
                    if len(pvalue_map) == 0:
                        break

                    bins_max, pvalue = max(pvalue_map.items(), key = lambda x:x[1])

                    if pvalue <= self.alpha_significance:
                        break
                    
                    # Remove bins
                    bins_chi = [i for i in bins_chi if not i in bins_max]
                
                    # Remove pvalues
                    keys = list(pvalue_map.keys())
                    for key in keys:
                        for b in key:
                            if b in bins_max:
                                pvalue_map.pop(key)
                                break

                    # Add composed bin
                    new_bin = ()
                    for i in bins_max:
                        new_bin += i
                    bins_chi.append(new_bin)

                    # Calculate pvalues
                    for bin_i in bins_chi:
                        if bin_i == new_bin or frozenset((bin_i,new_bin)) in pvalue_map.keys():
                            continue

                        n_i_0 = (y[np.isin(x,bin_i)]==0).sum()
                        n_i_1 = (y[np.isin(x,bin_i)]==1).sum()
                        n_j_0 = (y[np.isin(x,new_bin)]==0).sum()
                        n_j_1 = (y[np.isin(x,new_bin)]==1).sum()
                        
                        if ((n_i_0, n_i_1, n_j_0, n_j_1).count(0) >= 2
                            and not ((n_i_0 == 0 and n_i_1 > 0 and n_j_0 > 0 and n_j_1 == 0)
                                     or (n_i_0 > 0 and n_i_1 == 0 and n_j_0 == 0 and n_j_1 > 0))):
                            pvalue = 1
                        else:
                            pvalue = ss.chi2_contingency([[n_i_0, n_i_1],
                                                          [n_j_0, n_j_1]]).pvalue
                       
                        pvalue_map[frozenset((bin_i,new_bin))] = pvalue

                # Save bins
                for composed_bins in bins_chi:
                    for b in composed_bins:
                        bins_map['bins']['cat'] = bins_map['bins'].get('cat',dict())
                        bins_map['bins']['cat'][b] = idx
                    idx += 1

        #missing
        if not included_missing:
            for val in np.unique(x_missing):
                bins_map['bins']['missing'] = bins_map['bins'].get('missing',dict())
                bins_map['bins']['missing'][val] = idx #same index

        # Save composed bins
        self.exog_bins[xname] = self._compose_bins(bins_map)

        # Calculate stats
        self._calc_stats(x_arr,
                         y_arr,
                         per_arr,
                         xname,
                         idx_cont,
                         self._epsilon)


    def fit(self,X,y,per,reset=True,return_bins=False,disable_tqdm=False):
        '''
        Fit one or many variables

        Arguments
        _________
        X : pd.DataFrame
            An object that contains named arrays of exogenous variable observations.
            Can be mixed-type. Note that observations must not contain nan or inf values.

        y : np.array
            Array of endogenous variable observations, must be binary.
            Note that observations must not contain nan or inf values.

        per : np.array
            Array of string values that indicate observation time period.
            Note that observations must not contain nan or inf values.

        reset : bool
            TIDE stores fitted bins. If reset = True, then the storage will
            be emptied when fit() is called. Else, it will store
            existing variables, add new ones during the fitting process.
            Note that if new variable has the same name as existing one,
            the latter will be overwrited.
        '''
        assert isinstance(X,pd.DataFrame)
        assert y.sum() > 0 and y.shape[0] - y.sum() > 0
        assert X.shape[0] == y.shape[0] == per.shape[0]

        for col in X.columns:
            self._validate_naninf(X[col])

        self._validate_naninf(y)
        self._validate_naninf(per)

        if reset:
            self.exog_bins = dict()
            self.exog_woes = dict()
            self.stats = self.stats.drop(self.stats.index)
            self.stats_per = self.stats_per.drop(self.stats_per.index)

        progress = tqdm(X.columns, disable=disable_tqdm)
        for col in progress:
            progress.set_postfix_str(f'Fitting now: {col}')
            x = X[col].values
            self._fit_single(x,y,per,col)

        if return_bins:
            return self.exog_bins
        

    def _transform_single(self,x,xname,handle_unknown):
        '''Transform single variable'''
        composed_bins = self.exog_bins[xname]
        woes = self.exog_woes[xname]
        round_brackets = 'right' if composed_bins['trend'] =='pos' else 'left'

        # Split x into continuous, discrete and missing parts
        idx_cont, idx_cat, idx_missing = self.idx_from_mixed(x)

        # Iterate over bins and transform values
        x_woe = np.array(x,dtype='O',copy=True)

        for idx, atomic_bins in composed_bins['bins'].items():
            woe_i = woes[idx]
            for b, bintype in atomic_bins.items():
                if bintype == 'cont':
                    if round_brackets == 'right':
                        idx_bin = (x[idx_cont] >= b[0]) & (x[idx_cont] < b[1])
                    elif round_brackets == 'left':
                        idx_bin = (x[idx_cont] > b[0]) & (x[idx_cont] <= b[1])
                    else:
                        raise ValueError('Round brackets can be only "left" or "right"')
                    
                    x_woe[idx_cont] = np.where(idx_bin, woe_i, x_woe[idx_cont])
                    
                elif bintype in ('cat','missing'):
                    idx_bin = x == b
                    x_woe[idx_bin] = woe_i
                else:
                    raise ValueError(f'Bin type {bintype} is not supported')
                
        # Check for unchanged values and handle them
        idx_unchanged = x == x_woe            
        if np.any(idx_unchanged):

            if handle_unknown == 'keep':
                pass
            
            elif handle_unknown == 'worst':
                x_woe[idx_unchanged] = min(woes.values())
            
            elif handle_unknown == 'best':
                x_woe[idx_unchanged] = max(woes.values())
            
            elif handle_unknown in ('missing_worst', 'missing_best'):
                found_missing_bin = False
                idx_mw = -1
                for idx, atomic_bins in composed_bins['bins'].items():
                    for b, bintype in atomic_bins.items():
                        if bintype == 'missing':
                            found_missing_bin = True
                            idx_mw = idx
                    if found_missing_bin:
                        break
                if idx_mw >= 0:
                    woe_i = woes[idx_mw]
                else:
                    if handle_unknown == 'missing_worst':
                        woe_i = min(woes.values())
                    if handle_unknown == 'missing_best':
                        woe_i = max(woes.values())
                x_woe[idx_unchanged] = woe_i
                
            else:
                raise ValueError(f'handle_unknown {handle_unknown} is not supported')
            
            warnings.warn(f'There are values in variable "{xname}" that are not covered by fitted bins and treated as "{handle_unknown}" by WoE.')

        return x_woe


    def transform(self, X, xnames=None, handle_unknown='keep'):
        '''
        Transform variable using previously calculated bins
        Arguments
        _________
        X : pd.DataFrame | pd.Series | np.array
            Data to transform using fitted bins and WoE values
        xnames : array-like
            Column names
        handle_unknown : {'keep', 'worst', 'best', 'missing_worst', 'missing_best'} (default = 'keep')
            Specifies behavior on how to treat values that are not
            covered by bins. This usually happens to values that
            were not present in training sample when TIDE was fitted.
            Possible options:
            - 'keep': do not apply WoE transformation to such values
            - 'worst' : use lowest WoE already fitted
            - 'best' : use highest WoE already fitted
            - 'missing_worst' : try to find bin with missing values and use its WoE,
                if not found - use lowest WoE among all bins 
            - 'missing_best' : try to find bin with missing values and use its WoE,
                if not found - use highest WoE among all bins 

        Returns
        _______
        Transformed dataset
        '''
        
        # DataFrames
        if isinstance(X, pd.DataFrame):
            df_transformed = pd.DataFrame()
            # Check
            if not xnames:
                xnames = X.columns
            for name in xnames:
                if not name in self.exog_bins.keys():
                    raise KeyError(f'Variable {name} not found in exog_bins')
                x = X[name]
                self._validate_naninf(x)
            # Transform
            for name in xnames:
                x = X[name]
                df_transformed[f'{name}_woe'] = self._transform_single(x,name,handle_unknown)
            return df_transformed
        
        # Series
        if isinstance(X, pd.Series):
            name = X.name
            # Check
            if not name in self.exog_bins.keys():
                raise KeyError(f'Variable {name} not found in exog_bins')
            self._validate_naninf(X)
            # Transform
            return pd.Series(self._transform_single(X,name,handle_unknown), name=f'{name}_woe')
        
        if isinstance(X,np.ndarray):
            transformed = []
            # Check
            if len(X.shape) !=2:
                raise ValueError('X must have 2 dimensions')
            if not xnames:
                raise ValueError('Variable names are not specified')
            for i,name in enumerate(xnames):
                if not name in self.exog_bins.keys():
                    raise KeyError(f'Variable {name} not found in exog_bins')
                x = X[::,i]
                self._validate_naninf(x)
            # Transform
            for i,name in enumerate(xnames):
                x = X[::,i]
                transformed.append(self._transform_single(x,name,handle_unknown))
            return np.array(transformed).T


    def fit_transform(self, X, y, per, xnames=None, reset=True, return_bins=False, disable_tqdm=False, handle_unknown='keep'):
        '''Scikit-learn style fit and transform of one or many variables'''
        self.fit(X=X,y=y,per=per,reset=reset,return_bins=return_bins,disable_tqdm=disable_tqdm)
        transformed = self.transform(X=X,xnames=xnames,handle_unknown=handle_unknown)
        return transformed
    

    def _calc_stats(self,
                    x,
                    y,
                    per,
                    xname,
                    idx_cont = None,
                    epsilon = 1e-6):
        '''Calculates some stats and stores them in DataFrame'''

        # Get bins
        composed_bins = self.exog_bins[xname]
        n_bins = len(composed_bins['bins'])
        trend = composed_bins['trend']
        round_brackets = 'right' if trend == 'pos' else 'left'

        # Create string representation for bins
        composed_bins_str = []
        for atomic_bins in composed_bins['bins'].values():
            bin_str = []
            for b, bintype in atomic_bins.items():
                if bintype == 'cont':
                    if trend == 'pos':
                        bin_str.append(f'[{b[0]:.4f},{b[1]:.4f})')
                    else:
                        bin_str.append(f'({b[0]:.4f},{b[1]:.4f}]')
                elif bintype in ('cat','missing'):
                    bin_str.append(b)
                else:
                    raise ValueError(f'Bin type {bintype} is not supported')
            composed_bins_str.append('; '.join(bin_str))

        # Calculate stats across all dataset
        eventrates, events, totals = calc_eventrates(x,
                                                     y,
                                                     composed_bins['bins'],
                                                     idx_cont,
                                                     round_brackets,
                                                     return_counts=True)
        WoEs = calc_WoEs(x,y,composed_bins['bins'],idx_cont,round_brackets)
        IVs = calc_IVs(x,y,composed_bins['bins'],idx_cont,round_brackets)
        IV = np.nansum(IVs)

        # Write WoEs
        self.exog_woes[xname] = {idx:woe for idx,woe in zip(composed_bins['bins'].keys(),WoEs)}

        # Write stats
        stats_i = pd.DataFrame({'variable':[xname]*n_bins,
                                'trend':[trend]*n_bins,
                                'bin':composed_bins_str,
                                'n_obs':totals,
                                'n_events':events,
                                'event_rate':eventrates,
                                'WoE':WoEs,
                                'IV_contrib':IVs,
                                'IV_total':IV})
        if self.stats.empty:
            self.stats = stats_i
        else:
            self.stats = self.stats[self.stats['variable'] != xname]
            self.stats = pd.concat([self.stats,stats_i],
                                   ignore_index=True,
                                   axis=0).reset_index(drop=True)
            
        # Calculate stats for every period
        per_unique = np.unique(per)

        PSIs_seq = calc_PSIs(x,y,per,composed_bins['bins'],idx_cont,round_brackets,'sequential')
        PSIs_base = calc_PSIs(x,y,per,composed_bins['bins'],idx_cont,round_brackets,'on_base')

        for p, PSI_seq, PSI_base in zip(per_unique, PSIs_seq, PSIs_base):
            eventrates, events, totals = calc_eventrates(x[per==p],
                                                         y[per==p],
                                                         composed_bins['bins'],
                                                         idx_cont[per==p],
                                                         round_brackets,
                                                         return_counts=True)
            
            stats_per_i = pd.DataFrame({'period':[p] * n_bins,
                                        'variable':[xname] * n_bins,
                                        'trend':[trend] * n_bins,
                                        'bin':composed_bins_str,
                                        'n_obs':totals,
                                        'n_events':events,
                                        'event_rate':eventrates,
                                        'PSI_seq':[PSI_seq] * n_bins,
                                        'PSI_base':[PSI_base] * n_bins})
            if self.stats_per.empty:
                self.stats_per = stats_per_i
            else:
                self.stats_per[self.stats_per['variable']!=xname]
                self.stats_per = pd.concat([self.stats_per,stats_per_i],
                                           ignore_index=True,
                                           axis=0).reset_index(drop=True)
            

    def plot(self,xnames = [],figsize=(15,6),dpi=120):
        '''Matplotlib plot for variable binning results'''
        for name in xnames:
            if not name in self.exog_bins.keys():
                raise ValueError(f'Feature "{name}" not found in exog_bins')
        for name in xnames:
            stats_i = self.stats[self.stats['variable']==name].copy().reset_index(drop=True)
            stats_i['n_obs%'] = stats_i['n_obs'] / stats_i['n_obs'].sum()

            plt.rc('font', size=8)
            fig, ax = plt.subplots(1,3, figsize=figsize,dpi=dpi)
            iv = self.stats[self.stats["variable"]==name]["IV_total"].max()
            fig.suptitle(f'Binning results for {name}\nIV={iv:.4f}')

            # Result of binning on all periods
            ax[0].set_title('Bin stats')
            ax[0].bar(stats_i['bin'],stats_i['n_obs'],label='n_obs',color='lightblue')
            for i,row in stats_i.iterrows():
                ax[0].annotate(row['n_obs'],(i,row['n_obs']/2),ha='center',va='center')

            ax[0].set_xticks(stats_i['bin'])
            ax[0].set_xticklabels(stats_i['bin'],rotation=90)
            ax[0].set_xlabel('Bins')
            ax[0].set_ylabel('Bin size')

            ax_0_1 = ax[0].twinx()
            ax_0_1.plot(stats_i['bin'],stats_i['event_rate'],label='event_rate',color='darkblue',marker='.')
            for i,row in stats_i.iterrows():
                ax_0_1.annotate(f'{row["event_rate"]:.4f}',(i,row['event_rate']),ha='center')
            ax_0_1.set_ylabel('Event rate')

            lines, labels = ax[0].get_legend_handles_labels()
            lines2, labels2 = ax_0_1.get_legend_handles_labels()
            ax[0].legend(lines + lines2, labels + labels2, bbox_to_anchor=(0, 1), loc='lower right')
            
            # Event rate per bin change over time
            stats_per_i = self.stats_per[self.stats_per['variable']==name].copy()

            for b in stats_per_i['bin'].unique():
                ax[1].plot(stats_per_i[stats_per_i['bin']==b]['period'].astype('str'),
                           stats_per_i[stats_per_i['bin']==b]['event_rate'], marker = '.',label=b)
            ax[1].set_title('Event rate stability over time')
            ax[1].set_xlabel('Period')
            ax[1].set_ylabel('Event rate')
            ax[1].set_xticks(stats_per_i[stats_per_i['bin']==b]['period'].astype('str'))
            ax[1].set_xticklabels(stats_per_i[stats_per_i['bin']==b]['period'].astype('str'),rotation=90)
            ax[1].legend(title='Bins', bbox_to_anchor=(0.5, -0.4), loc='lower center')

            # PSI over time
            bottom = np.zeros(stats_per_i['period'].unique().shape[0])
            for b in stats_per_i['bin'].unique():
                labels = stats_per_i[stats_per_i['bin']==b]['period'].astype('str').values
                heights = (stats_per_i[stats_per_i['bin']==b]['n_obs'] / (stats_per_i.groupby('period').sum()['n_obs'].values + self._epsilon)).values
                ax[2].bar(labels, heights, width=0.975, bottom = bottom, label=b, alpha = 0.5)
                for i, val in enumerate(zip(heights,bottom + heights / 2)):
                    ax[2].annotate(f'{val[0]:.1%}',(i,val[1]),color='grey',ha='center',va='center')
                bottom = bottom + heights
            
            ax[2].set_title('Population Stability Index')
            ax[2].set_xlabel('Period')
            ax[2].set_ylabel('Bin size, %n_obs in period')
            ax[2].set_xticks(stats_per_i[stats_per_i['bin']==b]['period'].astype('str'))
            ax[2].set_xticklabels(stats_per_i[stats_per_i['bin']==b]['period'].astype('str'),rotation=90)
            
            ax_2_1 = ax[2].twinx()
            ax_2_1.plot(stats_per_i[stats_per_i['bin']==b]['period'].astype('str'),
                       stats_per_i[stats_per_i['bin']==b]['PSI_seq'], marker = '.', color = 'green', label = 'PSI_seq')
            
            for i,val in enumerate(stats_per_i[stats_per_i['bin']==b]['PSI_seq'].values):
                ax_2_1.annotate(f'{val:.3f}',(i,val),ha='center')

            ax_2_1.plot(stats_per_i[stats_per_i['bin']==b]['period'].astype('str'),
                       stats_per_i[stats_per_i['bin']==b]['PSI_base'], marker = '.', color = 'darkblue', label = 'PSI_base')
            
            for i,val in enumerate(stats_per_i[stats_per_i['bin']==b]['PSI_base'].values):
                ax_2_1.annotate(f'{val:.3f}',(i,val),ha='center')
            
            ax_2_1.set_ylim(0,np.clip(a = max(stats_per_i['PSI_seq'].fillna(0).max() * 1.1,
                                            stats_per_i['PSI_base'].fillna(0).max() * 1.1),
                                      a_min = 1,
                                      a_max = 1000))
            ax_2_1.set_ylabel('PSI')

            lines, labels = ax[2].get_legend_handles_labels()
            lines2, labels2 = ax_2_1.get_legend_handles_labels()
            ax[2].legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.5, -0.4), loc='lower center')
            
            fig.tight_layout()
            plt.show()