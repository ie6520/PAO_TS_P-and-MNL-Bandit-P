import cplex
import numpy as np

def getOptimalAssortment(n,w,r,B,log = True):
    #print(w)
    
    p = cplex.Cplex()
    if not log:
        p.set_log_stream(None)
        p.set_error_stream(None)
        p.set_warning_stream(None)
        p.set_results_stream(None)
        
    obj = [0]+r
    p.objective.set_sense(p.objective.sense.maximize)
    p.variables.add(obj = obj,names = ['x'+str(i) for i in range(n+1)],lb = [0]*(n+1))
    rows = []
    rowP = [1 for i in range(n+1)]
    rowC = [-B]+[1.0/w[i] for i in range(n)]
    va = [i for i in range(n+1)]
    rows = [[va,rowP],[va,rowC]]
    for i in range(1,n+1):
        temp = [0 for j in range(n+1)]
        temp[i] = 1/w[i-1] 
        temp[0] = -1
        rows.append([va,temp])
    rhs = [0 for i in range(n+2)]
    rhs[0] = 1
    senses = ['L']*(n+2)
    senses[0] = 'E'
    p.linear_constraints.add(lin_expr=rows,senses=senses,rhs = rhs)
    #p.write("lpex.lp")
    
    p.solve()
    #val = p.solution.get_objective_value()
    re = p.solution.get_values()
    
    #print(re)
    #print(val)
    
    x = []
    for i in range(1,n+1):
        if re[i]>1e-5: x.append(i-1)
    #print(re,w,"!")
    return x
    

if __name__=='__main__':
    print()