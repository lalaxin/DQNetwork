"""
考虑用户骑行成本约束，决定用户去或者不去临近区域还车，转化成背包问题，将用户的骑行成本作为重量，价值为1，即尽可能多的下一个时间段的用户骑到车
"""

from scipy import optimize as opt
import numpy as np
from scipy.optimize import minimize
import math
import copy
from km3maxdistance import km
# from helloword import User
# 动态规划
class solution:
    def ZeroOnePack(W, V, MW):  # 0-1背包
        # 存储最大价值的一维数组
        valuelist = [0] * (MW + 1)
        # 存储物品编号的字典
        codedict = {i: [] for i in range(0, MW + 1)}
        # 开始计算
        for ii in range(len(W)):  # 从第一个物品
            copyvalue = valuelist.copy()
            copydict = codedict.copy()
            for jj in range(MW + 1):  # 从重量0
                if jj >= W[ii]:  # 如果重量大于物品重量
                    copyvalue[jj] = max(valuelist[jj - W[ii]] + V[ii], valuelist[jj])  # 选中第ii个物品和不选中，取大的
                    # 输出被选中的物品编号
                    if copyvalue[jj] > valuelist[jj]:
                        copydict[jj] = [ii]
                        for hh in codedict[jj - W[ii]]:
                            copydict[jj].append(hh)
            codedict = copydict.copy()  # 更新
            valuelist = copyvalue.copy()  # 更新
        print('所需物品：', sorted([1 + code for code in codedict[MW]]))
        return '最大价值：', valuelist[-1]

    # print(ZeroOnePack_Simple(weight, value, maxweight))


class ulpkm:
    def __init__(self,user,region,pB,k,B,cell,celllength):
        self.cell=cell
        self.cellltngth=celllength
        self.user=user
        self.region=region
        self.pB=pB
        self.k=k
        self.B=B
        self.pi=np.zeros(len(user))
        self.kmuser=copy.deepcopy(user)
        initregion = copy.deepcopy(region)
        psokm = km(region=initregion, user=self.kmuser)
        psokm.build_graph()
        self.kmuser = psokm.KM()
        # print("self.kmuser",self.kmuser)
    # 计算两个区域之间的距离
    def objective(self,x):
        temp = 0
        for i in range(len(self.kmuser)):
            if (self.kmuser[i][5] != -1 and self.kmuser[i][6] != -1):
                temp += math.pow((x[2 * i]+1/2)*self.cellltngth - self.kmuser[i][5], 2) + math.pow((x[2 * i + 1]+1/2)*self.cellltngth - self.kmuser[i][6], 2)
        return temp

    def constraint1(self,x):
        for i in range(len(self.kmuser)):
            # 用户匹配缺车匹配成功
            if (self.kmuser[i][5] != -1 and self.kmuser[i][6] != -1):
                lslarr = math.sqrt(math.pow((self.user[i][0] - self.user[i][2]), 2) + math.pow(
                    (self.user[i][1] - self.user[i][3]), 2))  # 起点到终点
                lslact = math.sqrt(
                    math.pow((self.user[i][0] - x[2 * i]), 2) + math.pow((self.user[i][1] - x[2 * i + 1]), 2))  # 起点到实际位置
                larrlact = math.sqrt(
                    math.pow((x[2 * i] - self.user[i][2]), 2) + math.pow((x[2 * i + 1] - self.user[i][3]), 2))
                self.pi[i] = self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact)
                if (self.pi[i] < 0):
                    self.pi[i] = 0
            # 匹配未成功，用户无需匹配，直接归还到目的地
            else:
                self.pi[i] = 0
        return -(sum(self.pi) - self.B)

    def constraint2(self,x):
        return x-np.rint(x)

    def run(self):
        cons = list()
        # for i in range(len(self.user)):
        #     con = lambda x: -(math.pow(x[2 * i] - self.user[i][2], 2) + math.pow(x[2 * i + 1] - self.user[i][3], 2) - math.pow(
        #         self.user[i][4], 2))
        #     cons.append({'type': 'ineq', 'fun': con})
        cons.append({'type': 'ineq', 'fun': self.constraint1})
        cons.append({'type': 'eq', 'fun': self.constraint2})

        x0 = list()
        for i in range(len(self.user)):
            x0.append((self.user[i][2]-self.cellltngth/2)//self.cellltngth)
            x0.append((self.user[i][3]-self.cellltngth/2)//self.cellltngth)
        # print("x0",x0)
        bnds = [[0,self.cell-1 ] for x in x0]

        solution = minimize(self.objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x
        # print('目标值: ' + str(self.objective(x)))
        # print('答案为')
        # for i in range(len(x)):
        #     x[i]=round(x[i],3)
        # for i in range(len(self.user)):
        #     print('x' + str(i) + '=' + str((x[2 * i]+1/2)*self.cellltngth))
        #     print('y' + str(i) + '=' + str((x[2 * i+1]+1/2)*self.cellltngth))

        return x,self.objective(x)


# user=copy.deepcopy(User)
# Region=[[0, 2, 6, 1.0, 1.0], [0, 1, 10, 3.0, 1.0], [0, 1, 3, 1.0, 3.0], [0, 2, 9, 3.0, 3.0]]
# ulp1=ulpkm(user=user,region=Region,pB=1,k=2,B=5,cell=4,celllength=2)
# x,y=ulp1.run()
# print(x)
# print(y)
# Region=[[0, 0, 0, 1.5, 1.5], [0, 0, 0, 4.5, 1.5], [0, 0, 0, 7.5, 1.5], [0, 0, 1, 10.5, 1.5], [0, 1, 0, 1.5, 4.5], [0, 0, 1, 4.5, 4.5], [0, 0, 1, 7.5, 4.5], [0, 0, 0, 10.5, 4.5], [0, 0, 0, 1.5, 7.5], [0, 1, 1, 4.5, 7.5], [0, 0, 0, 7.5, 7.5], [0, 0, 2, 10.5, 7.5], [0, 0, 0, 1.5, 10.5], [0, 1, 1, 4.5, 10.5], [0, 0, 1, 7.5, 10.5], [0, 0, 2, 10.5, 10.5]]
# Region=[[0, 0, 0, 1.5, 1.5], [0, 0, 1, 4.5, 1.5], [0, 0, 0, 7.5, 1.5], [0, 0, 0, 10.5, 1.5], [0, 0, 1, 1.5, 4.5], [0, 0, 0, 4.5, 4.5], [0, 0, 0, 7.5, 4.5], [0, 0, 1, 10.5, 4.5], [0, 0, 0, 1.5, 7.5], [0, 0, 1, 4.5, 7.5], [0, 0, 2, 7.5, 7.5], [0, 0, 1, 10.5, 7.5], [0, 0, 3, 1.5, 10.5], [0, 0, 1, 4.5, 10.5], [0, 0, 0, 7.5, 10.5], [0, 0, 1, 10.5, 10.5]]
#
# user=[[6, 8, 4.5, 10.5, 2.2441861840702186, -1, -1], [12, 8, 4.5, 1.5, 3.0469537822133437, -1, -1], [1, 2, 1.5, 1.5, 3.826148690733884, -1, -1], [3, 4, 10.5, 4.5, 2.5208770865554477, -1, -1], [4, 5, 4.5, 4.5, 0.4784342320494666, -1, -1], [3, 9, 10.5, 10.5, 3.7265134398112427, -1, -1], [2, 9, 7.5, 10.5, 0.437954991661377, -1, -1], [0, 6, 1.5, 7.5, 3.637907100939602, -1, -1], [12, 2, 1.5, 4.5, 0.48169532108143925, -1, -1], [9, 12, 7.5, 1.5, 2.3972824075158976, -1, -1], [3, 9, 1.5, 4.5, 1.532545576551751, -1, -1], [4, 9, 7.5, 1.5, 1.9226308879732301, -1, -1]]
# # user=[[2, 2, 12, 2, 4.136562658805103, -1, -1], [5, 4, 1, 11, 2.1603046544320463, -1, -1], [11, 0, 12, 5, 3.448325332946857, -1, -1], [9, 12, 10, 8, 3.5310177934362983, -1, -1], [8, 2, 0, 11, 3.6208157554616824, -1, -1], [3, 4, 12, 1, 2.86471847777233, -1, -1], [7, 12, 6, 8, 1.050923237754301, -1, -1], [5, 2, 4, 7, 0.10251571186543765, -1, -1], [10, 6, 9, 0, 0.2617508493747883, -1, -1]]
# ulp1=ulpkm(user=user,region=Region,pB=1,k=100,B=2371,cell=4,celllength=3)
# ulp1.run()
