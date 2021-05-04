import numpy as np
import matplotlib.pyplot as plt

#from sample import Generate_theta
# import json

# with open("test.json", 'r') as f:
#     data = json.load(f)
#     reward_p = data["reward_p"]
#     reward_np = data["reward_np"]
#     reward_ora = data["reward_ora"]

def gen_figure(reward_np,reward_p,reward_ora):
    plt.clf()
    t = len(reward_p)
    x = list(range(t))

    regret_p = [sum(reward_ora[:i+1])-sum(reward_p[:i+1]) for i in range(t)]
    regret_np = [sum(reward_ora[:i+1])-sum(reward_np[:i+1]) for i in range(t)]

    #plt.plot(x,ratio,label="ratio",linestyle="-")

    plt.plot(x,regret_p,label="regret_p",linestyle="-")
    plt.plot(x,regret_np,label="regret_np",linestyle="-")
    plt.legend()
    # plt.show()
    # plt.savefig("result.png")
    return plt