import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape

import pymc3 as pm
from pymc3.model import ObservedRV
from theano.tensor.basic import dot,exp, mod,sum,concatenate,stack,transpose

def one_hot(s,N):
    one_hot_s = [0]*N
    for ss in s:
        one_hot_s[ss-1]=1
    return one_hot_s

def Generate_theta(x,s,I,N,K=1):
    #print(s,N)
    T = len(x)
    D = len(x[0])
    s = [one_hot(ss,N) for ss in s]
    # x = [[xt for _ in range(N)] for xt in x]
    x = np.array(x)
    s = np.array(s)

    model = pm.Model()
    with model:
        # Priors for unknown model parameters

        theta = pm.Normal("theta",mu=0,sigma=1,shape=(N,D))/np.sqrt(D)

        p_list = []
        for t in range(T):
            wt = dot(theta,x[t])
            swt = s[t]*wt
            sum_sw = sum(swt)
            p = exp(swt)/(1+sum_sw)
            p0 = 1/(1+sum_sw)
            p_list.append(concatenate(([p0],p)))

        I_obs = pm.Categorical("I_obs", p=stack(p_list,axis=0), observed=I)

    with model:
        step = pm.Metropolis()
        trace1 = pm.sample(tune = 2000,chains=1,step=step)

    return trace1["theta"][-1]

def Generate_theta_p(x,s,I,N,K,prod_f):
    #print(s,N,len(x))
    T = len(x)
    D = len(x[0])
    s = [one_hot(ss,N) for ss in s]
    # x = [[xt for _ in range(N)] for xt in x]
    x = np.array(x)
    s = np.array(s)

    model = pm.Model()
    with model:
        # Priors for unknown model parameters
        theta = pm.Normal("theta",mu=0,sigma=1,shape=(D,K))/np.sqrt(K*D)
        p_list = []
        for t in range(T):
            #print(prod_f)
            #print(np.transpose(theta))
            wt =  dot(dot(prod_f,transpose(theta)),x[t])
            swt = s[t]*wt
            sum_sw = sum(swt)
            p = exp(swt)/(1+sum_sw)
            p0 = 1/(1+sum_sw)
            p_list.append(concatenate(([p0],p)))

        I_obs = pm.Categorical("I_obs", p=stack(p_list,axis=0), observed=I)

    with model:
        step = pm.Metropolis()
        trace1 = pm.sample(tune = 2000,chains=1,step=step)

    return trace1["theta"][-1]

    
'''
D=2
N=3
H_ts = [[(1,1),(1,2,3),1],
        [(2,2),(1,2,3),1],
        [(2,2),(1,2,3),1],
        [(2,2),(1,2,3),1],
        [(2,2),(1,2,3),1]]
trace = Generate_theta(*zip(*H_ts),N)
'''

