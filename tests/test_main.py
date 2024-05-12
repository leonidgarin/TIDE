import numpy as np
import pandas as pd
import pytest
from tide.calc import calc_eventrates, calc_WoEs, calc_IVs, calc_PSIs
from tide import TIDE


def test_calc_eventrates():

    x = np.array([1,2]*1000)
    y = np.array([0,0,1,1]*500)
    composed_bins = {0:{(-np.inf, 2):'cont'},
                     1:{(2,+np.inf):'cont'}}
    eventrates_right = calc_eventrates(x,
                                      y,
                                      composed_bins,
                                      idx_cont=None,
                                      round_brackets='right',
                                      epsilon=1e-6,
                                      return_counts=False)
    assert np.all(np.isclose(np.array([0.5,0.5]),eventrates_right)) == True

    eventrates_left = calc_eventrates(x,
                                       y,
                                       composed_bins,
                                       idx_cont=None,
                                       round_brackets='left',
                                       epsilon=1e-6,
                                       return_counts=False)
    assert np.all(np.isclose(np.array([0.5,0.0]),eventrates_left)) == True


def test_calc_WoEs():
    np.random.seed(123)
    epsilon = 1e-6
    x = np.random.random(size=1000)
    g1 = np.random.choice([0,1],p=(0.2,0.8),size=1000)
    g2 = np.random.choice([0,1],p=(0.8,0.2),size=1000)
    y = np.where(x<0.5,g1,g2)

    n_events_all = y.sum()
    n_nonevents_all = (1-y).sum()

    n_events_bin1 = y[x<0.5].sum()
    n_nonevents_bin1 = (1-y[x<0.5]).sum()
    woe1 = np.log((n_nonevents_bin1/n_nonevents_all) / (epsilon + n_events_bin1/n_events_all))

    n_events_bin2 = y[x>=0.5].sum()
    n_nonevents_bin2 = (1-y[x>=0.5]).sum()
    woe2 = np.log((n_nonevents_bin2/n_nonevents_all) / (epsilon + n_events_bin2/n_events_all))

    composed_bins = {0:{(-np.inf, 0.5):'cont'},
                     1:{(0.5,+np.inf):'cont'}}
    
    woes = calc_WoEs(x,
                     y,
                     composed_bins,
                     idx_cont=None,
                     round_brackets='right',
                     epsilon=1e-6)
    
    assert np.all(np.isclose(np.array((woe1,woe2)), woes)) == True


def test_calc_IVs():
    np.random.seed(123)
    epsilon = 1e-6
    x = np.random.random(size=1000)
    g1 = np.random.choice([0,1],p=(0.2,0.8),size=1000)
    g2 = np.random.choice([0,1],p=(0.8,0.2),size=1000)
    y = np.where(x<0.5,g1,g2)

    n_events_all = y.sum()
    n_nonevents_all = (1-y).sum()

    n_events_bin1 = y[x<0.5].sum()
    n_nonevents_bin1 = (1-y[x<0.5]).sum()

    n_events_bin2 = y[x>=0.5].sum()
    n_nonevents_bin2 = (1-y[x>=0.5]).sum()

    iv1 = ((n_nonevents_bin1/n_nonevents_all) - (n_events_bin1/n_events_all)) * np.log((n_nonevents_bin1/n_nonevents_all) / (epsilon + n_events_bin1/n_events_all))
    iv2 = ((n_nonevents_bin2/n_nonevents_all) - (n_events_bin2/n_events_all)) * np.log((n_nonevents_bin2/n_nonevents_all) / (epsilon + n_events_bin2/n_events_all))

    composed_bins = {0:{(-np.inf, 0.5):'cont'},
                     1:{(0.5,+np.inf):'cont'}}
    
    iv = calc_IVs(x,
             y,
             composed_bins,
             idx_cont=None,
             round_brackets='right',
             epsilon=1e-6)
    
    assert np.all(np.isclose(np.array((iv1,iv2)),iv)) == True


def test_calc_PSIs_sequential():
    np.random.seed(123)
    size = 10000
    epsilon = 1e-6
    per = np.random.choice([1,2,3],size=size)
    x = np.random.random(size=size) * (1+per/10)
    y = np.random.choice([0,1],p=(0.2,0.8),size=size)

    n_p1_b1 = x[(per==1) & (x<0.5)].shape[0] / (epsilon + x[per==1].shape[0])
    n_p2_b1 = x[(per==2) & (x<0.5)].shape[0] / (epsilon + x[per==2].shape[0])
    n_p3_b1 = x[(per==3) & (x<0.5)].shape[0] / (epsilon + x[per==3].shape[0])
    n_p1_b2 = x[(per==1) & (x>=0.5)].shape[0] / (epsilon + x[per==1].shape[0])
    n_p2_b2 = x[(per==2) & (x>=0.5)].shape[0] / (epsilon + x[per==2].shape[0])
    n_p3_b2 = x[(per==3) & (x>=0.5)].shape[0] / (epsilon + x[per==3].shape[0])

    psi1 = (n_p2_b1 - n_p1_b1) * np.log(n_p2_b1 / (epsilon + n_p1_b1)) + (n_p2_b2 - n_p1_b2) * np.log(n_p2_b2 / (epsilon + n_p1_b2))
    psi2 = (n_p3_b1 - n_p2_b1) * np.log(n_p3_b1 / (epsilon + n_p2_b1)) + (n_p3_b2 - n_p2_b2) * np.log(n_p3_b2 / (epsilon + n_p2_b2))
    psi_check = np.array((np.nan, psi1, psi2))

    composed_bins = {0:{(-np.inf, 0.5):'cont'},
                     1:{(0.5,+np.inf):'cont'}}
    
    PSIs = calc_PSIs(x,
                     y,
                     per,
                     composed_bins,
                     idx_cont=None,
                     round_brackets='right',
                     how = 'sequential',
                     epsilon=1e-6)
    
    assert np.all(np.isclose(psi_check[1:], PSIs[1:])) == True
    assert np.isnan(PSIs[0]) == True


def test_df_cont_1per():
    size=10000
    np.random.seed(123)
    y = np.random.choice((0,1),p=(0.9,0.1),size=size)
    per = np.tile(0,size)
    x = np.where(y==1, np.random.normal(loc=0,scale=1,size=size), np.random.normal(loc=0,scale=1,size=size))
    df = pd.DataFrame({'x':x,'y':y,'per':per})
    mytide = TIDE(min_sample_rate = 0.05,
                    min_class_obs = 1,
                    min_bound = -np.inf,
                    max_bound = np.inf,
                    min_er_delta = 0,
                    n_prebins = 'log',
                    missing_categories = [],
                    missing_strategy = 'separate_bin',
                    missing_rate = 0.05,
                    cat_strategy = 'separate_bin',
                    alpha_significance = 0.05)
    bins = mytide.fit(df[['x']],df['y'],df['per'], return_bins = True)
    print(bins)
    assert bins == {'x': {'trend': 'pos',
                    'bins': {0: {(-np.inf, -1.4248190354707209): 'cont'},
                    1: {(-1.4248190354707209, 0.22620696557053013): 'cont'},
                    2: {(0.22620696557053013, 1.3991676705894323): 'cont'},
                    3: {(1.3991676705894323, np.inf): 'cont'}}}}
                        


