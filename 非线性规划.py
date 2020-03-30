from scipy import optimize as opt
import numpy as np
from scipy.optimize import minimize
import math

# 目标函数
def objective(x):
	return (x[0]-3)**2+(x[1]-1)**2

def constraint1(x):
    return -(math.sqrt((x[0]-4)**2+(x[1]-1)**2)-0.95*0.95) # 不等约束

def constraint2(x):
	return -(math.sqrt((x[0]-4)**2+(x[1]-4)**2)+2*((x[0]-4)**2)+2*(x[1]-1)-3.9)  # 不等约束

# 边界约束
b = (0.0, None)
bnds = (b, b)

con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
# con3 = {'type': 'eq', 'fun': constraint3}
# con4 = {'type': 'eq', 'fun': constraint4}
cons = ([con1, con2])  # 2个约束条件
x0 = np.array([0, 0])
# 计算
solution = minimize(objective, x0, method='SLSQP',  bounds=bnds, constraints=cons)
x = solution.x

print('目标值: ' + str(objective(x)))
print('答案为')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))


# ----------------------------------

