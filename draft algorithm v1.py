import numpy as np
import matplotlib.pyplot as plt
import math as mt

# MNL model
def MNL(z, S):
    global X
    prob = [0 for _ in range(len(S)+1)]
    cpv = [mt.exp(np.dot(X[S[i]],z)) for i in range(len(S))]
    base = sum(cpv)+1
    prob[0] = 1/base
    for i in range(1,len(prob)):
        prob[i] = cpv[i-1]/base
    return np.array(prob)

# the argmax to provide assortment St
def assortment_provider(z):
    global r_sorted, N_sorted
    baseline = MNL(z, [N_sorted[0]])[1]*r_sorted[0]
    for i in range(2,len(N)):
        potential_assortment = [N_sorted[l] for l in range(i)]
        proba = MNL(z, potential_assortment)
        new = sum([proba[j+1]*r_sorted[j] for j in range(i)])
        if new >= baseline:
            baseline = new
        else:
            break
    return np.array([N_sorted[m] for m in range(i-1)])

# Environment setup
#np.random.seed(100)
productcount = 60
productfeaturecount = 20
product_selected = ''
assortment_offered = ''
T = 200
revenue_per_round = [0 for i in range(T)]
revenue_per_round_optimal = [0 for i in range(T)]
total_revenue = np.array([0 for i in range(T)])
total_revenue_optimal = np.array([0 for i in range(T)])
R = 0
R_optimal = 0
N = list(range(productcount)) # set of all product
r = np.array([np.random.random() for _ in range(productcount)]) # revenue of products
r_sorted = r.tolist()
r_sorted.sort(reverse=True)
N_sorted = np.argsort(r)
N_sorted = N_sorted[::-1]

# Initialize the product feature matrix
#X = np.zeros((productcount,productfeaturecount))
#for i in range(X.shape[0]):
#    for j in range(X.shape[1]):
        #X[i,j] = np.random.randint(0,2)
        #X[i,j] = np.random.random()
X = (2*np.random.normal(0,1.0,size=(productcount,productfeaturecount))+2*np.random.uniform(size=(productcount,productfeaturecount)))/np.sqrt(productfeaturecount)
#X = np.random.uniform(0.5,1,size=(productcount,productfeaturecount))

# Model parameters setup
true_beta = (2*np.random.normal(0,1.0,size=(productfeaturecount))-np.random.uniform(size=(productfeaturecount)))/np.sqrt(productfeaturecount)
#true_beta = np.array(([(np.random.random()-0.5)*10 for _ in range(productfeaturecount)]))
beta = [0 for _ in range(productfeaturecount)]
Nk = np.zeros(productfeaturecount)

# Online dynamics
for t in range(T):

    # assortment by algorithm
    total = 0
    cumu = []
    if t == 0 or product_selected!='': # Exploitation
        St = assortment_provider(beta)
        assortment_offered = St
    else: # Exploration
        St = np.setdiff1d(N,assortment_offered)
        assortment_offered = St
    prob = MNL(true_beta, St)
    for i in range(len(prob)):
        total += prob[i]
        cumu.append(total)
    rand = np.random.random()
    for i in range(len(cumu)):
        if rand <= cumu[i]:
            if i == 0: # outside option chosen
                product_t = ''
                product_selected = ''
                for k in St:
                    for j in range(len(Nk)):
                        #if X[k, j] == 1:
                        #    Nk[j] -= 1
                        Nk[j] -= X[k,j]
                for j in range(len(beta)):
                    if Nk[j]>0:
                        beta[j] = np.log(Nk[j]*np.exp(1))
                    elif Nk[j]==0:
                        beta[j] = 0
                    else:
                        beta[j] = -np.log(-Nk[j]*np.exp(1))
                break
            else: # product_t chosen
                product_t = St[i-1]
                for j in range(len(Nk)):
                    #if X[product_t,j] == 1:
                    #    Nk[j] += 1
                    Nk[j] += X[product_t,j]
                for j in range(len(beta)):
                    if Nk[j]>0:
                        beta[j] = np.log(Nk[j]*np.exp(1))
                    elif Nk[j]==0:
                        beta[j] = 0
                    else:
                        beta[j] = -np.log(-Nk[j]*np.exp(1))
                product_selected = St[i - 1]
                revenue_per_round[t] = r[product_t]
                break

    # optimal assortment
    total = 0
    cumu = []
    St_optimal = assortment_provider(true_beta)
    prob_optimal = MNL(true_beta, St_optimal)
    for i in range(len(prob_optimal)):
        total += prob_optimal[i]
        cumu.append(total)
    rand = np.random.random()
    for i in range(len(cumu)):
        if rand <= cumu[i]:
            if i == 0: # outside option chosen
                product_t_optimal = ''
                break
            else: # product_t chosen
                product_t_optimal = St_optimal[i-1]
                revenue_per_round_optimal[t] = r[product_t_optimal]
                break

    R += revenue_per_round[t]
    total_revenue[t] = R
    R_optimal += revenue_per_round_optimal[t]
    total_revenue_optimal[t] = R_optimal
    regret = total_revenue_optimal - total_revenue

    # some printing
    print('********************************************')
    print('t = ', t)
    print("Assortment provided by algorithm: ", St)
    print("product chosen: ", product_t)
    print("Assortment provided by optimal policy: ", St_optimal)
    print("product chosen: ", product_t_optimal)
    print("beta: ", beta)
    #print(X[9])
    #print(r[9]*MNL(beta,N)[St.index(9)], r[19]*MNL(beta,N)[20])
    #print(r[9]*MNL(true_beta, N)[10], r[19]*MNL(true_beta, N)[20])
# plot revenue Rt vs time t
t = [i for i in range(T)]
#plt.plot(t, total_revenue, label = 'algorithm')
#plt.plot(t, total_revenue_optimal, label = 'Optimal')
print("true_beta: ", true_beta)
percentage = total_revenue[-1]/total_revenue_optimal[-1]
print("{:.0%}". format(percentage))
plt.plot(t, regret, label = 'Regret')
plt.legend()
plt.show()