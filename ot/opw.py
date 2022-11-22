# -*- coding: utf-8 -*-
"""OPW Solver for entropic regularized optimal transport with order preserving constraints.
"""

import warnings
import torch
import numpy as np
from ot.bregman import sinkhorn, sinkhorn2
from ot.utils import list_to_array

from .backend import get_backend
import math



def opw_sinkhorn(a, b, M,lambda1=50, lambda2=0.1, delta=1, method='sinkhorn', numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, **kwargs):
    """
    Solve the entropic regularization OPW and return the OT matrix

    Args:
        X (ndarray): view1
        Y (ndarray): view2
        lambda1 (int, optional): weight of first term. Defaults to 50.
        lambda2 (float, optional): weight of second term. Defaults to 0.1.
        delta (int, optional): _description_. Defaults to 1.

    Returns:
        ot_plan:  ot_plan is the transport plan
    """
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    reg = lambda2

    E, F = get_E_F(a.shape[0], b.shape[0], backend=nx)
    M_hat = M - lambda1 * E + lambda2 * (F / (2 * delta ** 2) + np.log(delta * nx.sqrt(2 * math.pi)))

    return sinkhorn(a, b, M_hat, reg, method=method, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, warn=warn, **kwargs)

def opw_sinkhorn2(a, b, M,lambda1=50, lambda2=0.1, delta=1, method='sinkhorn', numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, **kwargs):
    r"""
    Solve the entropic regularization OPW and return the loss

    Args:
        X (ndarray): view1
        Y (ndarray): view2
        lambda1 (int, optional): weight of first term. Defaults to 50.
        lambda2 (float, optional): weight of second term. Defaults to 0.1.
        delta (int, optional): _description_. Defaults to 1.

    Returns:
        distance: distance is the distance between views
    """

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    reg = lambda2

    E, F = get_E_F(a.shape[0], b.shape[0], backend=nx)
    M_hat = M - lambda1 * E + lambda2 * (F / (2 * delta ** 2) + nx.log(delta * nx.sqrt(2 * math.pi)))

    return sinkhorn2(a, b, M_hat, reg,method=method, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, warn=warn, **kwargs)

#-------------------------------PARTIAL-------------------------------

def POT_feature_2sides(a,b,D, m=None, nb_dummies=1,dummy_value = 0):
    nx = get_backend(D)
    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                        " than 0.")
    elif m > np.min((np.sum(a), np.sum(b))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                        " equal than min(|a|_1, |b|_1).")
    # import ipdb; ipdb.set_trace()
    b_extended = np.append(b, [(np.sum(a) - m) / nb_dummies] * nb_dummies)
    # b_extended = nx.concatenate(b, [(nx.sum(a) - m) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)

    b_extended = nx.from_numpy(b_extended)
    a_extended = nx.from_numpy(a_extended)
    
    D_extended = nx.ones((len(a_extended), len(b_extended)))*dummy_value
    D_extended[-nb_dummies:, -nb_dummies:] = nx.max(D) * 2
    D_extended[:len(a), :len(b)] = D
    return a_extended, b_extended, D_extended

def POT_feature_1side(a,b,D, m=0.8, nb_dummies=1,dummy_value=0):
    nx = get_backend(D)
    a = a*m
    '''drop on side b --> and dummpy point on side a'''
    a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
    D_extended = nx.ones((len(a_extended), len(b)))*dummy_value
    D_extended[:len(a), :len(b)] = D

    b = nx.from_numpy(b)
    a_extended = nx.from_numpy(a_extended)
    return a_extended, b,D_extended

def opw_partial_wasserstein(a, b, M, m = None, nb_dummies=1,dummy_value=0, drop_both_side = False , lambda1=50, lambda2=0.1, delta=1, method='sinkhorn', numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, **kwargs):
    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M)
    
    reg = lambda2
    E, F = get_E_F(a.shape[0], b.shape[0], backend=nx)
    M = M - lambda1 * E + lambda2 * (F / (2 * delta ** 2) + nx.log(delta * np.sqrt(2 * math.pi)))
    
    if drop_both_side:
        a,b,M = POT_feature_2sides(a,b,M,m,nb_dummies=nb_dummies,dummy_value=dummy_value)
    else:
        '''drop on side b --> and dummpy point on side a'''
        a,b,M = POT_feature_1side(a,b,M,m,nb_dummies=nb_dummies,dummy_value=dummy_value)

    return sinkhorn(a, b, M, reg, method=method, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, warn=warn, **kwargs)


def opw_partial_wasserstein2(a, b, M, m = None, nb_dummies=1,dummy_value=0 ,drop_both_side = False , lambda1=50, lambda2=0.1, delta=1, method='sinkhorn', numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, **kwargs):
    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M)
    
    reg = lambda2
    E, F = get_E_F(a.shape[0], b.shape[0], backend=nx)
    M = M - lambda1 * E + lambda2 * (F / (2 * delta ** 2) + nx.log(delta * np.sqrt(2 * math.pi)))
    
    if drop_both_side:
        a,b,M = POT_feature_2sides(a,b,M,m,nb_dummies=nb_dummies,dummy_value=dummy_value)
    else:
        '''drop on side b --> and dummpy point on side a'''
        a,b,M = POT_feature_1side(a,b,M,m,nb_dummies=nb_dummies,dummy_value=dummy_value)

    return sinkhorn2(a, b, M, reg, method=method, numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log, warn=warn, **kwargs)


def get_E_F(N, M, backend):
    nx = backend

    mid_para = np.sqrt((1/(N**2) + 1/(M**2)))

    a_n = nx.arange(start=1, stop=N+1)
    b_m = nx.arange(start=1, stop=M+1)
    row_col_matrix = nx.meshgrid(a_n, b_m)  #indexing xy
    row = row_col_matrix[0].T / N # row = (i+1)/N
    col = row_col_matrix[1].T / M # col = (j+1)/M

    l = nx.abs(row - col) / mid_para

    E =  1 / ((row - col) ** 2 + 1)
    F = l**2
    return E, F














