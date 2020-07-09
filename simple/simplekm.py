"""
用户的起始区域和终点区域都使用区域中心点坐标表示
"""
import operator

import numpy as np
import math

import pandas as pd

from simple.kmmatcher import KMMatcher

class KM():
    def __init__(self,region,user,cellength,B,pB,k):
        self.sumlackbike=0
        self.ep=1e-10
        self.region=region
        self.user=user
        self.usernum=len(self.user)
        self.cellength=cellength
        self.B=B
        self.pB=pB
        self.k=k
        # self.cellnum=cellnum

    # km最优匹配
    # 车匹配区域，左边表示车，相当于用户数，右边表示区域内缺车数（即缺的车数总和）
    # 生成的数组（用户横坐标，用户纵坐标，区域缺车横坐标，区域缺车纵坐标，权值（两者之间的距离1/d））
    def build_graph(self):
        for i in range(len(self.region)):
            if (self.region[i][2] > 0):
                self.sumlackbike += self.region[i][2]
        # usernnum行sumlackbike列
        self.adj = [[0 for i in range(int(self.sumlackbike))] for i in range(int(self.usernum))]
        for i in range(self.usernum):
            a = 0
            for j in range(len(self.region)):
                if (self.region[j][2] > 0):
                    for k in range(self.region[j][2]):
                        a += 1
                        # 计算用户目的地和各个缺车区域之间的距离
                        d = -math.sqrt(math.pow((self.user[i][2] - self.region[j][3]), 2) + math.pow((self.user[i][3] - self.region[j][4]), 2))
                        # 将大于用户最大步行距离
                        if(-d>self.cellength):
                            d=-np.inf
                        self.adj[i][a - 1] = [self.user[i][2], self.user[i][3], self.region[j][3], self.region[j][4], d]

        self.adj_matrix = [[0 for i in range(self.sumlackbike)] for i in range(self.usernum)]
        for i in range(self.usernum):
            for j in range(self.sumlackbike):
                self.adj_matrix[i][j] = self.adj[i][j][4]

        if(self.sumlackbike<self.usernum):
            self.adj_matrix=np.transpose(self.adj_matrix)

    def km(self):
        self.build_graph()
        matchweight=[]
        # weight=np.array(self.adj_matrix).astype(np.float32)
        weight=np.array(self.adj_matrix)
        matcher=KMMatcher(weight)
        # matcher.solve(verbose=True)
        distance,matchweight=matcher.solve(verbose=False)
        # print("distance",distance)
        # print("matchweight",matchweight)
        for i in range (int(len(matchweight)/2)):
            self.user[matchweight[2*i]][5]=self.adj[matchweight[2*i]][matchweight[2*i+1]][2]
            self.user[matchweight[2 * i]][6] = self.adj[matchweight[2 * i]][matchweight[2 * i + 1]][3]
        # print("分配预算前self.user",self.user)
        return self.user

    def backpack(self,W,V, MW):  # 0-1背包

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

    def finaluser(self):
        self.km()
        weight=[]
        userid=[]
        value=[]
        MW=self.B
        for i in range (len(self.user)):
            if(self.user[i][5]!=-1 and self.user[i][6]!=-1 ):
                lslarr = math.sqrt(math.pow((self.user[i][0] - self.user[i][2]), 2) + math.pow((self.user[i][1] - self.user[i][3]), 2))  # 起点到终点
                lslact = math.sqrt(math.pow((self.user[i][0] - self.user[i][5]), 2) + math.pow((self.user[i][1] - self.user[i][6]),2))  # 起点到实际位置
                larrlact = math.sqrt(math.pow((self.user[i][5] - self.user[i][2]), 2) + math.pow((self.user[i][6] - self.user[i][3]), 2))
                weight.append(self.pB * (lslact - lslarr) + self.k * (larrlact) * (larrlact))
                userid.append(i)
                value.append(1)
        for i in range (len(weight)):
            min_index,min_number=min(enumerate(weight), key=operator.itemgetter(1))
            MW-=min_number
            if(MW<0):
                break
            else:
                weight[min_index]=1e10
                userid.remove(min_index)
        for i in range(len(userid)):
            self.user[userid[i]][5]=-1
            self.user[userid[i]][6] = -1
        # print("weight",weight)
        # print("value",value)
        # print("userid",userid)
        # valuelist=self.backpack(weight,value,MW)
        # print(valuelist)
        # print("分配预算后：",self.user)
        return self.user



#
# cellength=2
# User=[[0,0,5,3,10,-1,-1],[0,0,5,3,10,-1,-1],[0,0,1,3,10,-1,-1],[0,0,1,3,10,-1,-1]]
# Region=[[0, 2, 0, 1.0, 1.0],[0, 2,1, 3.0, 1.0],[0, 2, 0, 5.0, 1.0],[0, 2, 0, 1.0, 3.0],[0, 2, 2, 3.0, 3.0],[0, 2, 0, 5.0, 3.0],[0, 2, 1, 1.0, 5.0],[0, 2, 0, 3.0, 5.0],[0, 2, 1, 5.0, 5.0]]
# kmtest=KM(Region,User,cellength,40,1,5)
# kmtest.finaluser()

