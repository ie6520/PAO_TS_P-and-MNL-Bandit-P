
import numpy as np
from numpy.lib.function_base import place
import Optimal_Assortment
import matplotlib.pyplot as plt

from sample import Generate_theta
import json

#feature dimension
D = 5
#item set
N = [i+1 for i in range(20)]
#cardinality constraint
B = 20

r = np.random.uniform(size=len(N))
#2d-array
Theta_g=(2*np.random.normal(0,1.0,size=(len(N),D))-np.random.uniform(size = (len(N),D)))/np.sqrt(D)



def Receive_x():
    return np.random.rand(D)

def Prod(x):
    N_x = []
    return N

def getOptimalAssortment(Theta,Nx,x):
    #print(Theta)
    nx = len(Nx)
    wx = [0]*nx
    rx = [0]*nx
    for i in range(nx):
        #print(Theta[Nx[i]-1])
        wx[i] = np.exp(np.dot(Theta[Nx[i]-1],x))
        rx[i] = r[Nx[i]-1]
    opt_as = Optimal_Assortment.getOptimalAssortment(n = nx, w = wx, r = rx, B=B, log = False)
    for i in range(len(opt_as)):
        opt_as[i] = Nx[opt_as[i]]
        
    return opt_as

def getProbability(ast,x):
    n = len(ast)
    wx = [0]*n
    for i in range(n):
        wx[i] = np.exp(np.dot(Theta_g[ast[i]-1],np.array(x)))
    
    wx = [1]+wx
    sum_wx = sum(wx)
    for i in range(len(wx)): 
        wx[i]/=sum_wx
    
    return wx

def getOptimalValue(ast,x):
    prob = getProbability(ast, x)
    sum = 0
    for i in range(len(ast)):
        sum+=r[ast[i]-1]*prob[i+1]
    return sum
        
def getCustomerPick(ast,x):
    prob = getProbability(ast, x)
    draw = np.random.choice([0]+ast,1,p=prob)
    return draw[0]



def PAO_TS(T,r):
    #history trajectory
    H_TS=[]
    reward = []
    reward_ora = []
    for t in range(1,T+1):
        x = Receive_x()
        Nx = Prod(x)
        if len(H_TS)==0:
            Theta_ts = np.random.normal(0,1.0,size=(len(N),D))
        else:
            Theta_ts = Generate_theta(*zip(*H_TS),len(N))
        
        opt_as_ts = getOptimalAssortment(Theta_ts, Nx, x)
        opt_as_ora = getOptimalAssortment(Theta_g, Nx, x)
        
        getOptimalValue(opt_as_ora, x)
        
        I_t = getCustomerPick(opt_as_ts,x)
        
        reward.append(getOptimalValue(opt_as_ts, x))
        reward_ora.append(getOptimalValue(opt_as_ora,x))
        
        H_TS.append([x,opt_as_ts,I_t])
        print(sum(reward)/sum(reward_ora))
    
    return reward,reward_ora


if __name__=='__main__':
    T = 50
    reward,reward_ora = PAO_TS(T,r)
    x = list(range(T))
    plt.plot(x,reward,label="reward",linestyle="-", marker="^")
    plt.plot(x,reward_ora,label="reward_ora",linestyle="-", marker="s")
    plt.show()

    res = {"reward":reward,"reward_ora":reward_ora}

    with open("test.json", 'w') as f:
        json.dump(res, f)
