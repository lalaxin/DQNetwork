from scipy import optimize as opt
import numpy as np
from scipy.optimize import minimize
import math
import copy
from km2 import km

user=[[4, 4, 4, 1, 0.9568767390288544, -1, -1]]
Region=[[0, 2, 6, 1.0, 1.0], [0, 1, 10, 3.0, 1.0], [0, 1, 3, 1.0, 3.0], [0, 2, 9, 3.0, 3.0]]
kmuser=copy.deepcopy(user)
initregion=copy.deepcopy(Region)
psokm = km(region=initregion, user=kmuser)
psokm.build_graph()
kmuser = psokm.KM()

pi=np.zeros(len(user))
pB=1
k=2
B=0.9

# 目标函数
def objective(x):
    temp = 0
    for i in range(len(kmuser)):
        if (kmuser[i][5] != -1 and kmuser[i][6] != -1):
            temp += math.pow(x[2 * i] - kmuser[i][5], 2) + math.pow(x[2 * i + 1] - kmuser[i][6], 2)
    return temp
	# return (x[0]-3)**2+(x[1]-1)**2+(x[2]-3)**2+(x[3]-1)**2

def constraint1(x):
    for i in range(len(kmuser)):
        # 用户匹配缺车匹配成功
        if (kmuser[i][5] != -1 and kmuser[i][6] != -1):
            lslarr = math.sqrt(math.pow((user[i][0] - user[i][2]), 2) + math.pow(
                (user[i][1] - user[i][3]), 2))  # 起点到终点
            lslact = math.sqrt(
                math.pow((user[i][0] - x[2 * i]), 2) + math.pow((user[i][1] - x[2 * i + 1]), 2))  # 起点到实际位置
            larrlact = math.sqrt(
                math.pow((x[2 * i] - user[i][2]), 2) + math.pow((x[2 * i + 1] - user[i][3]), 2))
            pi[i] = pB * (lslact - lslarr) + k * (larrlact) * (larrlact)
            if (pi[i] < 0):
                pi[i] = 0
        # 匹配未成功，用户无需匹配，直接归还到目的地
        else:
            pi[i] = 0
    return  -(sum(pi) - B)
    # return -(math.sqrt((x[0]-4)**2+(x[1]-1)**2)-0.95) # 不等约束

cons=list()
for i in range(len(user)):
    con = lambda x: -(math.pow(x[2 * i] - user[i][2], 2) + math.pow(x[2 * i + 1] - user[i][3],2) - math.pow(user[i][4], 2))
    cons.append({'type': 'ineq', 'fun': con})
cons.append({'type': 'ineq', 'fun': constraint1})

b=(0,4)
bnd=list()
for i in range (len(user)):
    bnd.append(b)

x0=np.zeros((len(user)))
solution = minimize(objective, x0, method='SLSQP',  bounds=bnd, constraints=cons)
x = solution.x

print('目标值: ' + str(objective(x)))
print('答案为')
for i in range (len(user)):
    print('x'+i+'='+x[2*i])
    print('y'+i+'='+x[2*i+1])


