from scipy import optimize as opt
import numpy as np
from scipy.optimize import minimize
import math
import copy
from km3maxdistance import km
# from helloword import User

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

    def objective(self,x):
        temp = 0
        for i in range(len(self.kmuser)):
            if (self.kmuser[i][5] != -1 and self.kmuser[i][6] != -1):
                temp += math.pow(x[2 * i] - self.kmuser[i][5], 2) + math.pow(x[2 * i + 1] - self.kmuser[i][6], 2)
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

    def run(self):
        cons = list()
        # for i in range(len(self.user)):
        #     con = lambda x: -(math.pow(x[2 * i] - self.user[i][2], 2) + math.pow(x[2 * i + 1] - self.user[i][3], 2) - math.pow(
        #         self.user[i][4], 2))
        #     cons.append({'type': 'ineq', 'fun': con})
        cons.append({'type': 'ineq', 'fun': self.constraint1})

        x0 = list()
        for i in range(len(self.user)):
            x0.append(self.user[i][2])
            x0.append(self.user[i][3])
        bnds = [[0,self.cell*self.cellltngth ] for x in x0]

        solution = minimize(self.objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x
        # print('目标值: ' + str(self.objective(x)))
        # print('答案为')
        # for i in range(len(x)):
        #     x[i]=round(x[i],3)
        # for i in range(len(self.user)):
        #     print('x' + str(i) + '=' + str(x[2 * i]))
        #     print('y' + str(i) + '=' + str(x[2 * i + 1]))

        return x,self.objective(x)


# user=copy.deepcopy(User)
# Region=[[0, 2, 6, 1.0, 1.0], [0, 1, 10, 3.0, 1.0], [0, 1, 3, 1.0, 3.0], [0, 2, 9, 3.0, 3.0]]
# ulp1=ulpkm(user=user,region=Region,pB=1,k=2,B=5,cell=4,celllength=2)
# x,y=ulp1.run()
# print(x)
# print(y)


