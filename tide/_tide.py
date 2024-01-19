'''
TIDE - Time Invariant Discretization Engine

Leonid Garin, 2023

TODO:
- Rewrite code for new IV, WoE, ER calc functions output - done
- Transform - done
- fit_transform - done
- Plots - done
- add composed bins functionality
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
                 max_bound = +np.inf,
                 n_prebins = 'log',
                 monotonic_strategy = 'simple',
                 missing_categories = [],
                 missing_strategy = 'separate_bin',
                 missing_rate = 0.05,
                 cat_strategy = 'separate_bin',
                 alpha_significance = 0.05
                 ):
        '''
        TIDE - Time Invariant Discretization Engine

        Weight-of-Evidence (WoE) based binning is an useful
        strategy to target-encode mixed-type exogenous variables
        in the task of binary classification. This allows to
        create robust models for regulatory purposes,
        for example credit scorecards.

        The TIDE binning is performed on arrays containing mixed-type
        points: integers, floats, strings. The numbers are
        treated as continuous part, while strings as categorical;
        some strings as missing values.

        TIDE provides and guarantees the following restrictions:
        - The trend of event rate in each subsequent bin
        must be monotonical;
        - This trend must remain the same on different time
        intervals within the training data;
        - A bin must be at least 5%* size of total data
        observations;
        - A bin must contain at least 1* observation of
        both classes
        - Missing values must be taken into account: placed in
        separate bin if there are sufficient cases, or joined
        to some other bin (the "worst" / the "best").
        * values can be adjusted.

        After bins calculation, one can perform WoE transformation
        of exogenous variables and use the result as factors for
        logistic regression.

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

        n_prebins : 'log' or int (int >= 2, default = 'log')
            Number of bins specified for prebinning algorithm,
            which is Quantile binning. If 'log', then the following
            formula is used:
            n_prebins = max(3,ceil(log(N_total)))

        monotonic_strategy : {'simple', 'chi-squared'} (default = 'simple')
            Specifies algorithm behavior regarding the monotonicity
            constraint during the process of boundary points selection.
            If 'simple', then the event rate in each subsequent bin
            will be monotonically increasing (decreasing), but
            similar bins can appear.
            If 'chi-squared', then every candidate bin will be
            tested for similarity with the previous one with
            chi-squared test. For small bins, exact Fisher test is used.

        missing_categories : list of str
            This list must contain categories that must be treated as missing.
            Note that np.nan is not allowed because it has unpredictable
            behavior (f.e. np.nan != np.nan).

        missing_strategy : {'separate_bin','worst_bin','best_bin'} (default = 'separate_bin')
            Defines the algorithm behavior on how to treat
            missing values.
            If 'separate_bin', then there will be special bin
            for missing values regardless of its size.
            Other strategies are not implemented yet.

        missing_rate : float (0.0 < float < 1.0) (default = 0.05)
            (Not implemented yet)
            If missing_rate is higher (>=) then specified,
            the algorithm will create a separate bin for missings.
            Else, the algorithm will create composed bin according to chosen strategy.
            For 'worst_bin' and ascending trend, missings will be included in last bin.
            For 'worst_bin' and descending trend, missing will be included in first bin.
            For 'best_bin' and ascending trend, missings will be included in first bin.
            For 'best_bin' and descending trend, missing will be included in last bin.

        cat_strategy : {'separate_bin','chi-squared'} (default = 'separate_bin')
            This option specifies how the algorithm handles the
            categorical bins. Note that these strategies will not
            ensure stability of categorical bins over time.
            If 'separate_bin', then they will be kept as is.
            If 'chi-squared', then the algorithm will loop merge bins
            using chi-squared test (see ChiMerge by R.Kerber, 1992).
            For small bins, exact Fisher test is used.

        alpha_significance : float, (0.0 < float < 0.5, default = 0.05)
            The specified threshold for p-value to reject
            the null hypothesis. Used for all chi-squared/Fisher
            tests if applicable.

        Methods
        _______

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
                                                 'PSI'])
        
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

        if n_prebins != 'log':
            assert n_prebins >= 2
        self.n_prebins = n_prebins

        assert monotonic_strategy in {'simple', 'chi-squared'}
        self.monotonic_strategy = monotonic_strategy

        assert isinstance(missing_categories,(list,tuple,set,np.ndarray)) and all([isinstance(i,self._acceptable_types_cat) for i in missing_categories])
        self.missing_categories = missing_categories

        assert missing_strategy in {'separate_bin',}
        self.missing_strategy = missing_strategy

        assert 0.0 < missing_rate < 1.0
        self.missing_rate = missing_rate

        assert cat_strategy in {'separate_bin','chi-squared'}
        self.cat_strategy = cat_strategy

        assert 0.0 < alpha_significance < 0.5
        self.alpha_significance = alpha_significance


    def _validate_naninf(self,x):
        if np.any(pd.isna(x)) or x[x==np.inf].shape[0]>0:
            raise ValueError('Input contains NaN and/or inf')


    def idx_from_mixed(self,x):
        '''
        Takes mixed-type column and return boolean indexes
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
                composed['bins'][idx] = composed.get(idx,dict())
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
        # Split x into continuous, discrete and missing parts
        idx_cont, idx_cat, idx_missing = self.idx_from_mixed(x)

        x_arr = np.array(x,dtype='O')
        y_arr = np.array(y)
        per_arr = np.array(per)
        per_unique = np.unique(per_arr)

        x_cont = x_arr[idx_cont]
        y_cont = y_arr[idx_cont]
        per_cont = per_arr[idx_cont]

        if np.any(idx_cont): 
            # Calculate size of pre-bins
            if self.n_prebins == 'log':
                n_prebins = max(3,int(np.ceil(np.log(len(x)))))
            else:
                n_prebins = self.n_prebins

            # Pre-binning for trend determination
            prebins_map = equal_size_binning_cont(x_cont, n_prebins)
            prebins = self._compose_bins(prebins_map)['bins']

            # Trend (a1 > 0 = positive, else negative; if no continous part, then positive)
            prebins_eventrates = calc_eventrates(x_cont,y_cont,prebins)
            a1, _ = np.polyfit(np.arange(n_prebins),prebins_eventrates,deg=1)
            if np.isclose(a1,0):
                a1 = 1
        else: #no continuous part
            a1 = 1

        if a1 > 0:
            trend = 'pos'
        else:
            trend = 'neg'

        # Calculate crosstab for sorted unique x (continuous part) and periods
        # Every period has three columns: count of observations,
        # count of events per unique x, count of non-events
        
        # According to the NumPy docs (see https://numpy.org/doc/stable/reference/generated/numpy.unique.html),
        # the np.unique function also sorts x
        # TODO: add logic with missing in first or last bin
        x_cont_unique, idx_x_cont_unique = np.unique(x_cont,return_inverse=True)

        crosstab = x_cont_unique

        for p_i in per_unique:
            p_counts = np.bincount(idx_x_cont_unique,weights = per_cont==p_i)
            crosstab = np.vstack((crosstab,p_counts))
            p_events = np.bincount(idx_x_cont_unique,weights = (per_cont==p_i)*y_cont)
            crosstab = np.vstack((crosstab,p_events))
            p_nonevents = p_counts - p_events
            crosstab = np.vstack((crosstab,p_nonevents))

        crosstab = crosstab.T

        # Main cycle
        idx_base = 0
        idx_bounds = []
        best_candidate = None

        while idx_base < x_cont_unique.shape[0]:

            # Calculate cumulative event rate for every period
            er_cumulative = crosstab[idx_base:, 2::3].cumsum(axis=0) / (self._epsilon + crosstab[idx_base:, 1::3].cumsum(axis=0))

            # Calculate right window function according to trend
            if trend == 'pos':
                er_right_window = np.minimum.accumulate(er_cumulative[::-1],axis=0)[::-1]
            else: #neg
                er_right_window = np.maximum.accumulate(er_cumulative[::-1],axis=0)[::-1]

            # Choose candidate points that provide all the conditions for continuous part:
            # Monotonicity for every period
            cand_monotone = np.all(er_cumulative == er_right_window, axis=1)

            # Candidate bin has equal or more than min_sample_rate * 100% observations out of total
            cand_binsize = np.all((crosstab[idx_base:, 1::3].cumsum(axis=0) /
                                   (self._epsilon + crosstab[:, 1::3].sum(axis=0))) >= self.min_sample_rate, axis=1)

            # Every class is present with at least n=min_class_obs observations
            cand_classprecense = np.all(crosstab[idx_base:, 2::3].cumsum(axis=0) >= self.min_class_obs, axis=1) \
                                 & np.all(crosstab[idx_base:, 3::3].cumsum(axis=0) >= self.min_class_obs, axis=1)
            
            # TODO: add Chi-squared for previous bin

            # Choose candidate point
            best_candidate = np.argmax(np.all((cand_monotone,cand_binsize,cand_classprecense),axis=0))

            # Accept candidate point, if the remaining space satisfies the conditions of min_sample_rate and min_class_obs
            if crosstab[idx_base+best_candidate:].shape[0] > 0:
                flag_remaining_binsize = np.any(np.all((crosstab[idx_base+best_candidate:, 1::3].cumsum(axis=0) /
                                            (self._epsilon + crosstab[:, 1::3].sum(axis=0))) >= self.min_sample_rate, axis=1))
                flag_remaining_classprecense = np.any(np.all(crosstab[idx_base+best_candidate:, 2::3].cumsum(axis=0) >= self.min_class_obs, axis=1) \
                                        & np.all(crosstab[idx_base+best_candidate:, 3::3].cumsum(axis=0) >= self.min_class_obs, axis=1))
            else:
                flag_remaining_binsize = False
                flag_remaining_classprecense = False

            # Save point and update base index 
            if best_candidate and flag_remaining_binsize and flag_remaining_classprecense:
                idx_bounds.append(idx_base+best_candidate)
                if trend == 'pos':
                    idx_base += best_candidate
                else:
                    idx_base += best_candidate + 1
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

        cont_bounds = np.r_[min_bound, x_cont_unique[idx_bounds], max_bound]

        # Handle categorical bins
        bins_cat = np.unique(x[idx_cat])

        # Handle missing bins
        bins_missing = np.unique(x[idx_missing])

        # Map for bins
        bins_map = {'trend':trend, 'bins':dict()}

        idx = 0
        for val in zip(cont_bounds[:-1],cont_bounds[1:]):
            bins_map['bins']['cont'] = bins_map['bins'].get('cont',dict())
            bins_map['bins']['cont'][val] = idx
            idx += 1
        for val in bins_cat:
            bins_map['bins']['cat'] = bins_map['bins'].get('cat',dict())
            bins_map['bins']['cat'][val] = idx
            idx += 1
        for val in bins_missing:
            bins_map['bins']['missing'] = bins_map['bins'].get('missing',dict())
            bins_map['bins']['missing'][val] = idx #same index

        # Save bins
        self.exog_bins[xname] = self._compose_bins(bins_map)

        # Calculate stats

        self._calc_stats(x_arr,
                         y_arr,
                         per_arr,
                         xname,
                         idx_cont)



    def fit(self,X,y,per,reset=True,return_bins=False,disable_tqdm=False):
        '''
        Fit one or many variables

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

        TODO: Add multiprocessing
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

        progress = tqdm(X.columns, disable=disable_tqdm)
        for col in progress:
            progress.set_postfix_str(f'Fitting now: {col}')
            x = X[col].values
            self._fit_single(x,y,per,col)

        if return_bins:
            return self.exog_bins
        


    def _transform_single(self,x,xname):
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
                
        # Check for unchanged values                
        if np.any(x == x_woe):
            warnings.warn(f'There are values in variable "{xname}" that remained unchanged after WoE transformation. Check if your bins cover all the possible values.')

        return x_woe



    def transform(self,X, xnames=None):
        '''Transform variable using previously calculated bins'''
        
        # DataFrames
        if isinstance(X, pd.DataFrame):
            df_transformed = pd.DataFrame()
            # Check
            if not xnames:
                xnames = X.columns
            for name in xnames:
                if not name in self.exog_bins.keys():
                    raise KeyError('Variable {name} not found in exog_bins')
                x = X[name]
                self._validate_naninf(x)
            # Transform
            for name in xnames:
                x = X[name]
                df_transformed[f'{name}_woe'] = self._transform_single(x,name)
            return df_transformed
        
        # Series
        if isinstance(X, pd.Series):
            name = X.name
            # Check
            if not name in self.exog_bins.keys():
                raise KeyError('Variable {name} not found in exog_bins')
            self._validate_naninf(X)
            # Transform
            return pd.Series(self._transform_single(X,name), name=f'{name}_woe')
        
        if isinstance(X,np.ndarray):
            transformed = []
            # Check
            if len(X.shape) !=2:
                raise ValueError('X must have 2 dimensions')
            if not xnames:
                raise ValueError('Variable names are not specified')
            for i,name in enumerate(xnames):
                if not name in self.exog_bins.keys():
                    raise KeyError('Variable {name} not found in exog_bins')
                x = X[::,i]
                self._validate_naninf(x)
            # Transform
            for i,name in enumerate(xnames):
                x = X[::,i]
                transformed.append(self._transform_single(x,name))
            return np.array(transformed).T
            


    def fit_transform(self,X,y,per,reset=True,return_bins=False):
        '''Scikit-learn style'''
        self.fit(X,y,per,reset,return_bins)
        transformed = self.transform(X)
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
        # TODO: First and last brackets should be corrected according to bin
        composed_bins_str = []
        for atomic_bins in composed_bins['bins'].values():
            bin_str = []
            for b, bintype in atomic_bins.items():
                if bintype == 'cont':
                    if trend == 'pos':
                        bin_str.append(f'[{b[0]:.4f},{b[1]:.4f})')
                    elif trend == 'neg':
                        bin_str.append(f'({b[0]:.4f},{b[1]:.4f}]')
                    else:
                        raise ValueError(f'Trend {trend} is not supported')
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
        IV = IVs.sum()

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

        PSIs = calc_PSIs(x,y,per,composed_bins['bins'],idx_cont,round_brackets)

        for p,PSI in zip(per_unique,PSIs):
            eventrates, events, totals = calc_eventrates(x[per==p],
                                                         y[per==p],
                                                         composed_bins['bins'],
                                                         idx_cont[per==p],
                                                         round_brackets,
                                                         return_counts=True)
            
            stats_per_i = pd.DataFrame({'period':[p]*n_bins,
                                        'variable':[xname]*n_bins,
                                        'trend':[trend]*n_bins,
                                        'bin':composed_bins_str,
                                        'n_obs':totals,
                                        'n_events':events,
                                        'event_rate':eventrates,
                                        'PSI':[PSI]*n_bins})
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
            fig.suptitle(f'Binning results for {name}')

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
            ax_0_1.plot(stats_i['bin'],stats_i['event_rate'],label='event_rate',color='blue',marker='.')
            for i,row in stats_i.iterrows():
                ax_0_1.annotate(f'{row["event_rate"]:.4f}',(i,row['event_rate']),ha='center')
            ax_0_1.set_ylabel('Event rate')

            lines, labels = ax[0].get_legend_handles_labels()
            lines2, labels2 = ax_0_1.get_legend_handles_labels()
            ax[0].legend(lines + lines2, labels + labels2, bbox_to_anchor=(0, 1), loc='center right')
            
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
                       stats_per_i[stats_per_i['bin']==b]['PSI'], marker = '.', color = 'green', label = 'PSI')
            
            for i,val in enumerate(stats_per_i[stats_per_i['bin']==b]['PSI'].values):
                ax_2_1.annotate(f'{val:.3f}',(i,val),ha='center')
            
            ax_2_1.set_ylim(0,np.clip(stats_per_i['PSI'].fillna(0).max(), a_min=1,a_max=1000))
            ax_2_1.set_ylabel('PSI')

            lines, labels = ax[2].get_legend_handles_labels()
            lines2, labels2 = ax_2_1.get_legend_handles_labels()
            ax[2].legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.5, -0.4), loc='lower center')
            
            fig.tight_layout()
            plt.show()